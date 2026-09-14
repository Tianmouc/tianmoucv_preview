#include "tianmoucv.h"
#include "isp.h"
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

/*
WARNING:DEV vesion.
author: yihan lin , taoyi wang
deeply relied on 20240106 version tianmoucsdk
*/


namespace {

constexpr size_t kConeFrameBytes = static_cast<size_t>(16 + 320 * 160) * sizeof(int);
constexpr size_t kMaxRodFrameBytes = static_cast<size_t>(44448) * sizeof(int);

struct FrameWorkspace {
    std::mutex readMutex;
    cv::Mat timediff = cv::Mat::zeros(ROD_H, ROD_W, CV_8SC1);
    cv::Mat spatioDiffl = cv::Mat::zeros(ROD_H, ROD_W, CV_8SC1);
    cv::Mat spatioDiffr = cv::Mat::zeros(ROD_H, ROD_W, CV_8SC1);
    cv::Mat coneRaw = cv::Mat::zeros(CONE_H, CONE_W * 2, CV_16SC1);
    cv::Mat coneCompleted = cv::Mat::zeros(CONE_H, CONE_W * 2, CV_16SC1);
    cv::Mat coneUnsigned = cv::Mat::zeros(CONE_H, CONE_W * 2, CV_16UC1);
    cv::Mat coneDemosaic = cv::Mat::zeros(CONE_H, CONE_W * 2, CV_16UC3);
    cv::Mat rgb = cv::Mat::zeros(CONE_H, CONE_W * 2, CV_32FC3);
    cv::Mat rodPreviewLeft = cv::Mat::zeros(ROD_H, ROD_W * 2, CV_8SC1);
    cv::Mat rodPreviewRight = cv::Mat::zeros(ROD_H, ROD_W * 2, CV_8SC1);
    cv::Mat rodPreview = cv::Mat::zeros(ROD_H, ROD_W * 2, CV_8UC1);
    cv::Mat conePreviewRgb = cv::Mat::zeros(CONE_H, CONE_W * 2, CV_8UC3);
    cv::Mat conePreviewBgr = cv::Mat::zeros(CONE_H, CONE_W * 2, CV_8UC3);
};

std::mutex gWorkspaceMutex;
std::unordered_map<uint64_t, std::shared_ptr<FrameWorkspace>> gWorkspaces;

std::shared_ptr<FrameWorkspace> getWorkspace(uint64_t cameraHandle)
{
    std::lock_guard<std::mutex> lock(gWorkspaceMutex);
    auto& workspace = gWorkspaces[cameraHandle];
    if (!workspace)
        workspace = std::make_shared<FrameWorkspace>();
    return workspace;
}

std::shared_ptr<FrameWorkspace> findWorkspace(uint64_t cameraHandle)
{
    std::lock_guard<std::mutex> lock(gWorkspaceMutex);
    const auto found = gWorkspaces.find(cameraHandle);
    return found == gWorkspaces.end() ? nullptr : found->second;
}

void eraseWorkspace(uint64_t cameraHandle)
{
    std::lock_guard<std::mutex> lock(gWorkspaceMutex);
    gWorkspaces.erase(cameraHandle);
}

// Equivalent to the old sparse 5x5 filter, but performs four additions per
// pixel instead of invoking a general 25-tap convolution for four non-zero taps.
void completeConeRaw(const cv::Mat& raw, cv::Mat& completed)
{
    cv::parallel_for_(cv::Range(0, raw.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const int16_t* center = raw.ptr<int16_t>(y);
            const int16_t* up = y >= 2 ? raw.ptr<int16_t>(y - 2) : nullptr;
            const int16_t* down = y + 2 < raw.rows ? raw.ptr<int16_t>(y + 2) : nullptr;
            int16_t* output = completed.ptr<int16_t>(y);
            for (int x = 0; x < raw.cols; ++x) {
                int neighborSum = 0;
                if (up) neighborSum += up[x];
                if (down) neighborSum += down[x];
                if (x >= 2) neighborSum += center[x - 2];
                if (x + 2 < raw.cols) neighborSum += center[x + 2];
                output[x] = cv::saturate_cast<int16_t>(
                    static_cast<int>(center[x]) + cvRound(neighborSum * 0.25f));
            }
        }
    });
}

void processCone(int* frameData, FrameWorkspace& workspace)
{
    tianmouc::process::cone_reader(frameData, workspace.coneRaw, CONE_W, CONE_H);
    completeConeRaw(workspace.coneRaw, workspace.coneCompleted);
    workspace.coneCompleted.convertTo(workspace.coneUnsigned, CV_16UC1);
    cv::cvtColor(workspace.coneUnsigned, workspace.coneDemosaic, cv::COLOR_BayerRG2RGB);
    workspace.coneDemosaic.convertTo(workspace.rgb, CV_32FC3, 0.25);

    // Preserve the original gray-world AWB result without split/merge buffers.
    const cv::Scalar average = cv::mean(workspace.rgb);
    const double target = (average[0] + average[1] + average[2]) / 3.0;
    const cv::Vec3f gain(
        average[0] > 1e-12 ? static_cast<float>(target / average[0]) : 1.0f,
        average[1] > 1e-12 ? static_cast<float>(target / average[1]) : 1.0f,
        average[2] > 1e-12 ? static_cast<float>(target / average[2]) : 1.0f);
    workspace.rgb.forEach<cv::Vec3f>([gain](cv::Vec3f& pixel, const int*) {
        pixel[0] *= gain[0];
        pixel[1] *= gain[1];
        pixel[2] *= gain[2];
    });
}

class FrameReturnGuard {
public:
    FrameReturnGuard(lynFrame_t& frame, lynCameraHandle_t cameraHandle)
        : frame_(frame), cameraHandle_(cameraHandle) {}

    ~FrameReturnGuard()
    {
        if (frame_.data != nullptr)
            lynPutFrame(&frame_, cameraHandle_);
    }

private:
    lynFrame_t& frame_;
    lynCameraHandle_t cameraHandle_;
};

bool getFrameImpl(uint64_t cameraHandle, tianmoucData* metadata, bool latest)
{
    if (cameraHandle == 0 || metadata == nullptr)
        return false;

    const auto workspace = findWorkspace(cameraHandle);
    if (!workspace)
        return false;
    std::lock_guard<std::mutex> readLock(workspace->readMutex);

    metadata->td_p = workspace->timediff.ptr<uchar>();
    metadata->sdl_p = workspace->spatioDiffl.ptr<uchar>();
    metadata->sdr_p = workspace->spatioDiffr.ptr<uchar>();
    metadata->rgb_p = workspace->rgb.ptr<float>();

    lynFrame_t cameraFrame{};
    const auto nativeHandle = reinterpret_cast<lynCameraHandle_t>(cameraHandle);
    const bool successful = latest
        ? lynGetLatestFrame(&cameraFrame, nativeHandle)
        : lynGetFrame(&cameraFrame, nativeHandle);
    if (!successful)
        return false;

    FrameReturnGuard returnFrame(cameraFrame, nativeHandle);
    if (cameraFrame.data == nullptr)
        return false;

    try {
        int* frameData = static_cast<int*>(cameraFrame.data);
        if (cameraFrame.dataType == LYN_CONE) {
            if (cameraFrame.length < kConeFrameBytes)
                return false;
            processCone(frameData, *workspace);
            metadata->isRod = 0;
            metadata->timestamp = tianmouc::process::get_cone_timestamp(frameData);
            return true;
        }

        if (cameraFrame.dataType == LYN_ROD) {
            if (cameraFrame.length < 16 * sizeof(int) ||
                cameraFrame.length > kMaxRodFrameBytes ||
                cameraFrame.length % sizeof(int) != 0) {
                return false;
            }
            workspace->timediff.setTo(0);
            workspace->spatioDiffl.setTo(0);
            workspace->spatioDiffr.setTo(0);
            const int rodSize = static_cast<int>(cameraFrame.length / sizeof(int));
            const int decodeStatus = tianmouc::process::rod_decoder(
                frameData, workspace->timediff, workspace->spatioDiffl,
                workspace->spatioDiffr, ROD_W, ROD_H, rodSize);
            if (decodeStatus < 0)
                return false;
            metadata->isRod = 1;
            metadata->timestamp = tianmouc::process::get_rod_timestamp(frameData);
            return true;
        }
    } catch (const cv::Exception& error) {
        std::cerr << "[cpp sdk] frame processing failed: " << error.what() << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "[cpp sdk] frame processing failed: " << error.what() << std::endl;
    }
    return false;
}

}  // namespace

uint64_t tmOpenCamera(int device_id)
{
    const int deviceCount = lynCameraEnumerate();
    if (device_id < 0 || device_id >= deviceCount) {
        std::cerr << "[cpp sdk] invalid camera id " << device_id
                  << "; available devices: " << deviceCount << std::endl;
        return 0;
    }

    lynCameraHandle_t cameraHandle = nullptr;
    lynOpenCamera(&cameraHandle);
    if (cameraHandle == nullptr || !lynCameraInit(cameraHandle, device_id)) {
        if (cameraHandle != nullptr)
            lynCameraUninit(cameraHandle);
        return 0;
    }
    if (!lynCameraConfigFPGAMode(cameraHandle, 8, 3)) {
        lynCameraUninit(cameraHandle);
        return 0;
    }

    const uint64_t handle = reinterpret_cast<uint64_t>(cameraHandle);
    getWorkspace(handle);
    return handle;
}

bool tmCameraUninit(uint64_t cameraHandle)
{
    if (cameraHandle == 0)
        return false;
    eraseWorkspace(cameraHandle);
    return lynCameraUninit(reinterpret_cast<lynCameraHandle_t>(cameraHandle));
}

bool tmStartTransfer(uint64_t cameraHandle)
{
    return cameraHandle != 0 &&
           lynStartRecvData(reinterpret_cast<lynCameraHandle_t>(cameraHandle));
}

bool tmGetFrame(uint64_t cameraHandle, tianmoucData* metadata)
{
    return getFrameImpl(cameraHandle, metadata, false);
}

bool tmGetLatestFrame(uint64_t cameraHandle, tianmoucData* metadata)
{
    return getFrameImpl(cameraHandle, metadata, true);
}

bool tmRenderRodPreview(uint64_t cameraHandle, uchar* output, int outputBytes)
{
    constexpr int kPreviewBytes = ROD_H * ROD_W * 2;
    if (cameraHandle == 0 || output == nullptr || outputBytes < kPreviewBytes)
        return false;

    const auto workspace = findWorkspace(cameraHandle);
    if (!workspace)
        return false;
    std::lock_guard<std::mutex> readLock(workspace->readMutex);
    cv::resize(workspace->spatioDiffl, workspace->rodPreviewLeft,
               cv::Size(ROD_W * 2, ROD_H), 0, 0, cv::INTER_NEAREST);
    cv::resize(workspace->spatioDiffr, workspace->rodPreviewRight,
               cv::Size(ROD_W * 2, ROD_H), 0, 0, cv::INTER_NEAREST);
    tianmouc::isp::spatioReconstruct(
        workspace->rodPreviewLeft, workspace->rodPreviewRight,
        workspace->rodPreview);
    std::memcpy(output, workspace->rodPreview.data, kPreviewBytes);
    return true;
}

bool tmRenderConePreview(uint64_t cameraHandle, uchar* output, int outputBytes)
{
    constexpr int kPreviewBytes = CONE_H * CONE_W * 2 * 3;
    if (cameraHandle == 0 || output == nullptr || outputBytes < kPreviewBytes)
        return false;

    const auto workspace = findWorkspace(cameraHandle);
    if (!workspace)
        return false;
    std::lock_guard<std::mutex> readLock(workspace->readMutex);
    workspace->rgb.convertTo(workspace->conePreviewRgb, CV_8UC3);
    cv::cvtColor(workspace->conePreviewRgb, workspace->conePreviewBgr,
                 cv::COLOR_RGB2BGR);
    std::memcpy(output, workspace->conePreviewBgr.data, kPreviewBytes);
    return true;
}


/************************************
(5) free the struct 

void freeTianmoucData(tianmoucData* metadata){
    if(metadata){
        if(metadata->td_p)  free(metadata->td_p);
        if(metadata->sdl_p) free(metadata->sdl_p);
        if(metadata->sdr_p) free(metadata->sdr_p);
        if(metadata->rgb_p) free(metadata->rgb_p);
    }
}
************************************/


bool tmExposureSet(uint64_t cameraHandle, int rodAEtime, int coneAEtime,
                   int rodGainV, int coneGainV, int RODADCprecision,
                   int RODINTmode, int CONEINTmode){
    if (cameraHandle == 0)
        return false;
    const auto nativeHandle = reinterpret_cast<lynCameraHandle_t>(cameraHandle);
    
    bool conestat = lynCameraConfigSensorConeExp(nativeHandle, coneAEtime);
    
    conestat = lynCameraConfigSensorConeAnaGain(nativeHandle, coneGainV) && conestat;

    uint interface = ((RODINTmode << 1) & 0x3) | (CONEINTmode & 0x3);

    bool rodstat = lynCameraConfigSensorRodExp(nativeHandle, rodAEtime, RODADCprecision, interface);
    
    rodstat = lynCameraConfigSensorRodAnaGain(nativeHandle, rodGainV) && rodstat;

    return rodstat && conestat;

}

bool IICconfig_download(uint64_t cameraHandle, const char* IICconfigPath){
    if (cameraHandle == 0 || IICconfigPath == nullptr)
        return false;

    try {
        int RODADCprecision = 8;
        int RODINTmode  = 1;
        int CONEINTmode = 1;
        uint sensor_interface = ((RODINTmode << 1) & 0x3) | (CONEINTmode & 0x3);
        const auto nativeHandle = reinterpret_cast<lynCameraHandle_t>(cameraHandle);

        bool status = lynCameraConfigFPGAMode(nativeHandle, RODADCprecision, sensor_interface);

        status = lynCameraConfigSensorFull(nativeHandle, IICconfigPath) && status;

        status = lynCameraConfigFPGAMode(nativeHandle, RODADCprecision, sensor_interface) && status;

        return status;

    }catch (const std::exception& e) {
        std::cerr << "[cpp sdk] configuration failed: " << e.what() << std::endl;

    }
    return false;
}


#ifdef TIANMOUCV_BUILD_APP
int main(int argc, char* argv[])
{   //Camera handle
    tianmoucData metadata;

    cv::Mat Ix = cv::Mat::zeros(ROD_H, ROD_W, CV_32FC1);
    cv::Mat Iy = cv::Mat::zeros(ROD_H, ROD_W, CV_32FC1);
    std::cout << "[cpp sdk]before open " << std::endl;

    uint64_t cameraHandle = tmOpenCamera();

    std::cout << "[cpp sdk]after open device cameraHandle:" <<(lynCameraHandle_t)cameraHandle<<std::endl;

    IICconfig_download(cameraHandle,"./lib/Golden_HCG_seq.csv");

    tmExposureSet(cameraHandle, 1200, 15000, 1, 1, 8,1,1);//don't change last three parameters for Golden_HCG_seq.csv

    tmStartTransfer(cameraHandle);

    int render_gap = 5;
    int i = 0;
    while(1){
        i ++;
        tmGetFrame(cameraHandle,&metadata);
        std::cout<<"[cpp sdk]frame:"<<i<<std::endl;
        std::cout<<"[cpp sdk]time:"<<metadata.timestamp<<"0us"<<std::endl;
        if(metadata.isRod==1){
            std::cout<<"[cpp sdk]get rod"<<std::endl;
            if ((i%render_gap) == 0){
                cv::Mat SD_recon = cv::Mat::zeros(ROD_H, ROD_W, CV_32F);
                cv::Mat SDL(ROD_H, ROD_W, CV_8SC1, metadata.sdl_p);
                cv::Mat SDR(ROD_H, ROD_W, CV_8SC1, metadata.sdr_p);
                tianmouc::isp::SDlSDR_2_SDxSDy(SDL,SDR,Ix,Iy);
                int itter = 5;//control the blender's diffussion range
                tianmouc::isp::poisson_blend(SD_recon, Ix, Iy, itter);
                SD_recon.convertTo(SD_recon,CV_8UC1);
                cv::resize(SD_recon, SD_recon, cv::Size(ROD_W*2, ROD_H), 0, 0, cv::INTER_NEAREST);
                cv::imshow("SD-Reconstruction", SD_recon);
                cv::waitKey(1);
            }
        }else{
            std::cout<<"[cpp sdk]get cone"<<std::endl;
            cv::Mat image(CONE_H, CONE_W*2, CV_32FC3, metadata.rgb_p);
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            cv::imshow("RGB", image/255.0);
            cv::waitKey(1);
        }
    }
    tmCameraUninit(cameraHandle);
    return 0;
}
#endif
