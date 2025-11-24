 #include "tianmoucv.h"
#include "isp.h"
#include "unistd.h"

/*
WARNING:DEV vesion.
author: yihan lin , taoyi wang
deeply relied on 20240106 version tianmoucsdk
*/


/************************************
(1) open a camera and init camera handle
************************************/
uint64_t  tmOpenCamera(int device_id) {

    lynCameraHandle_t cameraHandle=nullptr;

    int dev = lynCameraEnumerate();

    std::cout <<"[cpp sdk]get dev:" << dev<<std::endl;
    
    if(dev>0){
        lynOpenCamera(&cameraHandle);

        if(device_id>=dev){
            std::cout << "[cpp sdk]illegal device id " << device_id << " in " << dev<<std::endl;
        }
        
        lynCameraInit(cameraHandle, device_id);
        std::cout << "[cpp sdk]first inited " << dev << " handle:"<<(lynCameraHandle_t)cameraHandle<<std::endl;
        lynCameraConfigFPGAMode(cameraHandle, 8, 3);
        return (uint64_t)cameraHandle;
    }else{
        //cameraHandle = nullptr;
        std::cout << "[cpp sdk]bad inited " << dev << std::endl;
        return 0;
    }    
}

/************************************
(2) stop camera
************************************/
void tmCameraUninit(uint64_t  cameraHandle){
    lynCameraUninit((lynCameraHandle_t)cameraHandle);
}

/************************************
(3) stop camera
************************************/
void tmStartTransfer(uint64_t  cameraHandle){
    lynStartRecvData((lynCameraHandle_t)cameraHandle);//打开数据接收、开了多个线程
    std::cout << "[cpp sdk]tianmouc StartRecv Data " << std::endl;
}

/************************************
(4) open a camera and init camera handle 
************************************/

int _FIRST_INIT_BUFFER = true;
bool tmGetFrame(uint64_t  cameraHandle, tianmoucData* metadata){

    lynFrame_t cam_frames;
    int MAX_FAILED_COUNT  = 10000;
    int faliedCount = 0;

    cv::Mat timediff = cv::Mat::zeros(ROD_H, ROD_W, CV_8SC1);
    cv::Mat spatioDiffl = cv::Mat::zeros(ROD_H, ROD_W, CV_8SC1);
    cv::Mat spatioDiffr = cv::Mat::zeros(ROD_H, ROD_W, CV_8SC1);
    cv::Mat raw = cv::Mat::zeros(CONE_H, CONE_W*2, CV_16SC1);
    cv::Mat demosaicRaw;
    std::vector<cv::Mat> dst;
    cv::Mat processed;

    TimeRecorder recorder;

    if(_FIRST_INIT_BUFFER){
        metadata->rgb_p = (float*)malloc(sizeof(float)*CONE_W*CONE_H*2*3);
        metadata->td_p = (uchar*)malloc(sizeof(uchar)*ROD_W*ROD_H);
        metadata->sdl_p = (uchar*)malloc(sizeof(uchar)*ROD_W*ROD_H);
        metadata->sdr_p = (uchar*)malloc(sizeof(uchar)*ROD_W*ROD_H);
        _FIRST_INIT_BUFFER = false;
    }

    int* cone_pvalue = (int*)malloc(0x1A000 * 2 * sizeof(int));
    int* rod_pvalue = (int*)malloc(44448 * sizeof(int));

    //read something
    while(1){
        bool successful =lynGetFrame(&cam_frames, (lynCameraHandle_t)cameraHandle);
        if (successful) {
            faliedCount  = 0;
            if (cam_frames.dataType == LYN_CONE) {
                //recorder.start();
                if(DEBUG)std::cout<<"[cpp sdk]Cone"<<std::endl;
                int cone_effDataLength = 16 + 320 * 160;
                memcpy(cone_pvalue,(int*)cam_frames.data, sizeof(int)*cone_effDataLength);
                lynPutFrame(&cam_frames, (lynCameraHandle_t)cameraHandle);
                metadata->isRod = 0;
                //decode rgb
                int err_code = tianmouc::process::cone_reader(cone_pvalue,raw,CONE_W,CONE_H);
                uint64_t timestamp = tianmouc::process::get_cone_timestamp(cone_pvalue);
                int cone_now_cnt = tianmouc::process::get_cone_counter(cone_pvalue);
                //demosacing
                tianmouc::isp::RGBNormalizer norm(1,0,0,0,1,1,1);
                bool status = tianmouc::isp::prepocess(raw, CONE_W,CONE_H,dst,demosaicRaw, norm,true);
                cv::merge(dst,processed);

                metadata->timestamp = timestamp;
                memcpy(metadata->rgb_p,(float*)processed.data,sizeof(float)*CONE_H*CONE_W*2*3);
            }else if (cam_frames.dataType == LYN_ROD) {
                //recorder.start();
                metadata->isRod = 1;
                int length = cam_frames.length;
                int rod_effDataLength = length / 4;
                int* data_print = (int*)cam_frames.data;
                if(DEBUG)std::cout<<"[cpp sdk]Rod"<<std::endl;
                if(length > 147168){
                    if(WARNING)printf("\n[cpp sdk] Overflow! length %d, data0 %x\n", length, data_print[0]);
                    break;
                }
                memcpy(rod_pvalue, (int*)cam_frames.data, sizeof(int) * rod_effDataLength);
                lynPutFrame(&cam_frames, (lynCameraHandle_t)cameraHandle);
                int rodsize = length / 4;
                uint64_t timestamp = tianmouc::process::get_rod_timestamp(rod_pvalue);
                int rod_now_cnt = tianmouc::process::get_rod_counter(rod_pvalue);
                int code = tianmouc::process::rod_decoder((int*)rod_pvalue, timediff,spatioDiffl, spatioDiffr, ROD_W, ROD_H, rodsize);

                metadata->timestamp = timestamp;

                memcpy(metadata->td_p,timediff.data,sizeof(uchar)*ROD_W*ROD_H);
                memcpy(metadata->sdl_p,spatioDiffl.data,sizeof(uchar)*ROD_W*ROD_H);
                memcpy(metadata->sdr_p,spatioDiffr.data,sizeof(uchar)*ROD_W*ROD_H);
                //recorder.stop("[cpp sdk] aop latency:");
            }
            break;
        }
        else{
            usleep(50);
            faliedCount += 1;
            if(DEBUG && faliedCount%1000 == 0){
                std::cout<<"[cpp sdk]waiting for next frame.., failed times:"<<faliedCount<<std::endl;
            }
            if (faliedCount > MAX_FAILED_COUNT){
                std::cout<<"[cpp sdk]break";
                break;
            }
        }
    }
    if(cone_pvalue) free(cone_pvalue);
    if(rod_pvalue) free(rod_pvalue);
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


void tmExposureSet(uint64_t  cameraHandle, int rodAEtime, int coneAEtime, int rodGainV, int coneGainV,int RODADCprecision,int RODINTmode,int CONEINTmode){
    
    bool conestat = lynCameraConfigSensorConeExp((lynCameraHandle_t)cameraHandle, coneAEtime);
    
    conestat = lynCameraConfigSensorConeAnaGain((lynCameraHandle_t)cameraHandle, coneGainV);

    uint interface = ((RODINTmode << 1) & 0x3) | (CONEINTmode & 0x3);

    bool rodstat = lynCameraConfigSensorRodExp((lynCameraHandle_t)cameraHandle, rodAEtime,RODADCprecision ,interface);
    
    rodstat = lynCameraConfigSensorRodAnaGain((lynCameraHandle_t)cameraHandle, rodGainV);

    std::cout << "[cpp sdk] rodstat:"<<rodstat<< ",conestat:"<<conestat<< std::endl;

}

void IICconfig_download(uint64_t  cameraHandle, const char* IICconfigPath){

    try {
        int RODADCprecision = 8;
        int RODINTmode  = 1;
        int CONEINTmode = 1;
        int cameraConfigMode = 1;

        float max_rod_exp_time = 1240;

        std::cout<<IICconfigPath<<std::endl;


        uint sensor_interface = ((RODINTmode << 1) & 0x3) | (CONEINTmode & 0x3);

        lynCameraConfigFPGAMode((lynCameraHandle_t)cameraHandle, RODADCprecision, sensor_interface);

        bool status = lynCameraConfigSensorFull((lynCameraHandle_t)cameraHandle, IICconfigPath);

        lynCameraConfigFPGAMode((lynCameraHandle_t)cameraHandle, RODADCprecision, sensor_interface);

        std::cout << "[cpp sdk] config handle:"<<(lynCameraHandle_t)cameraHandle<<std::endl;

    }catch (exception& e) {
           std::cout<<"[cpp sdk]exception:"+std::string(e.what())<<std::endl;

    }
    return;
}


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
