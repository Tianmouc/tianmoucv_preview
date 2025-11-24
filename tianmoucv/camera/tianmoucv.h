#ifndef _TIANMOUC_H
#define _TIANMOUC_H 1

#include <iostream>
#include <fstream>
#include <chrono>
#include <time.h>
#include <vector>
#include <cstdint>
#include <chrono>
#include <stdint.h>
#include "tmc_proc.h"
#include "lyn_cam.h"

using namespace std;

#define ROD_W 160
#define ROD_H 160
#define CONE_W 320
#define CONE_H 320

#ifndef WARNING
#define WARNING true
#endif

#ifndef DEBUG
#define DEBUG false
#endif



class TimeRecorder {
private:
    std::chrono::steady_clock::time_point startTime;

public:
    void start() {
        startTime = std::chrono::steady_clock::now();
    }

    void stop(string NOTE) {
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << NOTE << duration << " ms." << std::endl;
    }
};


extern "C"{

struct tianmoucData{
    int isRod;
    uint64_t timestamp; 
    uchar * sdl_p;
    uchar * sdr_p;
    uchar * td_p;
    float * rgb_p;
};

uint64_t tmOpenCamera(int device_id =0);

void tmStartTransfer(uint64_t cameraHandle);

bool tmGetFrame(uint64_t cameraHandle,tianmoucData* metadata);

void tmCameraUninit(uint64_t cameraHandle);

void tmExposureSet(uint64_t cameraHandle, int rodAEtime, int coneAEtime, int rodGainV=1, int coneGainV=1,int RODADCprecision = 8,int RODINTmode=1,int CONEINTmode=1);

/************************************
void freeTianmoucData(tianmoucData* metadata);
************************************/

void IICconfig_download(uint64_t cameraHandle, const char* IICconfigPath);

}
#endif

