
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // 添加此头文件以支持std::vector和py::list的转换
//just for test
#include <filesystem>
namespace fs = std::filesystem;
//#include <experimental/filesystem>
//namespace fs = std::experimental::filesystem;

//#define DUMP
#define PKT_SIZE_MAX 38400//36800
//error and success code
#define DECODE_SUCCESS 0
#define ERR_ROW_ADDR_LOSS 1
#define ERR_WRONG_ADC_PREC 2
/************   Variable definations common **************/
#define FRM_HEAD_OFFSET 16
#define FRM_HEAD_CNT_TS_OFFSET 4
const unsigned int frm_head_cnt_ts_offset = FRM_HEAD_CNT_TS_OFFSET;
#define FRM_HEAD_ADC_PREC_OFFSET 0
#define FRM_HEAD_FrmCount_OFFSET 3
#define FRM_HEAD_TimeStampMSB_OFFSET 1
#define FRM_HEAD_TimeStampLSB_OFFSET 2
#define FRM_HEAD_READOUT_FLAG_OFFSET 0
#define FRM_HEAD_RAND_PATTERN_OFFSET 12
#define CONE_FRM_HEAD_TimeStampMSB_OFFSET 1
#define CONE_FRM_HEAD_TimeStampLSB_OFFSET 2
#define CONE_FRM_HEAD_FrmCount_OFFSET 3

/************   Variable definations for single camera! **************/
#define ROD_8B_ONE_FRM 0x9e00       // 158KB * 1024 / 4;//0x9e00


    //last frame address in DDR
#define ROD_4B_ONE_FRM 0x4D00 //

//define ROD_2B_ONE_FRM 0x1C40
//#define ROD_2B_LAST_FRM_ADDR 0x27fffcb00
#define ROD_2B_ONE_FRM 0x1D00

void usb_header_parse(int *pvalue, uint64_t * timestamp, int* fcnt, int* radc_prec){
    uint64_t ts_h = static_cast<uint64_t> (pvalue[1] & 0xffffff);
    uint64_t ts_m = static_cast<uint64_t> (pvalue[2] & 0xffffff);
    uint64_t ts_l = static_cast<uint64_t> (pvalue[3] & 0xffffff);
    *timestamp = static_cast<uint64_t> ((ts_h) << 48) + (ts_m << 24) + ts_l;
    uint32_t cnt_h = static_cast<uint32_t> (pvalue[4] & 0xffffff);
    uint32_t cnt_l = static_cast<uint32_t> (pvalue[5] & 0xffffff);
    *fcnt = static_cast<uint32_t> (cnt_h << 24) + cnt_l;
    unsigned int adc_bit_prec = (unsigned int)pvalue[0] & 0xf;
    *radc_prec = adc_bit_prec;

    return;
}

// -------------
// pure C++ code
// -------------

/**
 * @brief Key function of rod decoder, decode the tdiff and sdiff stream in pvalue
 * 
 * @param pvalue readout stream (unknown effective size)
 * @param temp_diff 160 * 160 temporal diff image
 * @param spat_diff_left  160 * 160 spatial diff (two left direction)  image
 * @param spat_diff_right 160 * 160 spatial diff (two right direction)  image
 * @param height image heiht, 160 for full array
 * @param width  image width, 160 for full array
 * @return total size in bits
 */
//decode one frame
int rod_decoder(int *pvalue, int8_t* temp_diff, int8_t* spat_diff_left, int8_t* spat_diff_right, int height, int width){
    
    //adc_bit_prec in frame head
    unsigned int adc_bit_prec = (unsigned int)pvalue[FRM_HEAD_ADC_PREC_OFFSET] & 0xf;
    //uint64_t timestamp_high =  (unsigned int)pvalue[FRM_HEAD_TimeStampMSB_OFFSET];
    //uint64_t timestamp_low = (unsigned int)pvalue[FRM_HEAD_TimeStampLSB_OFFSET];
    //uint64_t timestamp = ((uint64_t)timestamp_high << 32) + (uint64_t)timestamp_low;
     // set the total pixel number in one group packet
    uint8_t pix_per_grp = 0;
    // set the data mask
    uint16_t d_mask = 0;
    //unsigned int pkt_max_num;
    if(adc_bit_prec == 8){
        pix_per_grp = 2;
        d_mask = 0xff;
     //   pkt_max_num = ROD_8B_ONE_FRM;
    } else if(adc_bit_prec == 4){
        pix_per_grp = 4;
        d_mask = 0xf;
      //  pkt_max_num = ROD_4B_ONE_FRM;
    }
    else if(adc_bit_prec == 2){
        pix_per_grp = 12;
        d_mask = 0x3;
    //    pkt_max_num = ROD_2B_ONE_FRM;
    }else if (adc_bit_prec == 0){
        adc_bit_prec = 8;
      pix_per_grp = 2;
        d_mask = 0xff;      
    } 
    else{
        printf("Wrong ADC precision, %d\n", adc_bit_prec);
       // exit(-1);
        return ERR_WRONG_ADC_PREC;
    }
    int index = FRM_HEAD_OFFSET;
    //uint16_t width_ui16 = (uint16_t)
    int packet_tot_size = 0;
   // for(int ii = 0; ii < pkt_max_num; ii++){
    while(true){
        unsigned int pkt = (unsigned int)pvalue[index];//read a packt date from pvalue
        packet_tot_size++;
        //Detect the end of current frame
        if((pkt == 0xffffffff) || (pkt == 0x0))
            break;
        // adc bit prec = 8 or 4, share the same logic
        if(adc_bit_prec == 8 || adc_bit_prec == 4){
            pkt = pkt >> 8;
            //bool g_pkt_flag = ;
            if (pkt >> 19 == 0x1){
                unsigned int frm_cnt = pkt & 0x1ff;
                //printf("FRM %u read from DMA\n", frm_cnt);
            }
            else {
                if(pkt >> 23){
                    printf("The Row address packet may be lost !\n");
                    return ERR_ROW_ADDR_LOSS;
                }//else if(pkt == 0xffffffff)
                else {
                    uint16_t row_pkt_head = pkt >> 16;
                    //if(row_pkt_head == 0x10){
                    uint16_t eff_grp_num = (pkt >> 9 ) & 0x7f;
                    uint16_t row_addr = pkt & 0xff;
                    //Record group data to temp_diff, spat_diff_left and spat_diff_right frame 
                    uint16_t yaddr = width * row_addr;
                    for(int gi = 0; gi < eff_grp_num; gi++){
                        index++;
                        unsigned int grp_pkt = (unsigned int)pvalue[index];
                        packet_tot_size++;
                        //if((grp_pkt == 0xffffffff) ||  (grp_pkt == 0x0))
                        //    printf("Wrong group packet, now at last packet");
                        grp_pkt = grp_pkt >> 8;                        
                       /* if((grp_pkt >> 23) == 0){ //group packet flag
                            printf("Wrond group address\n");
                        }*/
                        if(grp_pkt >> 23){
                            uint16_t g_addr = (grp_pkt >> 16) & 0x7f;
                            uint16_t g_data = grp_pkt & 0xffff;
                            //uint16_t xaddr_base = ;
                            //uint16_t pixel_addr = yaddr + g_addr * pix_per_grp;
                            for(uint8_t pi = 0; pi < pix_per_grp; pi++){
                                uint16_t pixel_addr = yaddr + g_addr * pix_per_grp + pi;
                                //pixel_addr += pi;//100us
                                int8_t pix_data = (int8_t) ((g_data >> (adc_bit_prec * (pix_per_grp - 1 - pi))) & d_mask);//100us
                                if(adc_bit_prec == 4){
                                    pix_data = pix_data << 4;
                                }
                                //BYTE pix_data = (BYTE) ((g_data >> ((pix_per_grp - 1 - pi) << 3)) & d_mask);
                                //BYTE pix_data = (BYTE) ((g_data >> (8 * (2 - 1 - pi))) & d_mask);
                                // 230us for save to corresponding buffer
                                if(row_pkt_head == 0x10){ // TempDiff Row address packet flag
                                    temp_diff[pixel_addr] = pix_data;
                                } else if(row_pkt_head == 0x14){ // Spat Diff left Row address packet flag
                                    spat_diff_left[pixel_addr] = pix_data;
                                } else if (row_pkt_head == 0x16){ // Spat Diff right Row address packet flag
                                    spat_diff_right[pixel_addr] = pix_data;
                                } else{
                                    printf("BAD row packt head!\n");
                                }/**/
                            } 
                        }
                    }
                }
            }
            index++;
            //packet_tot_size++;
        } 
        // adc bit prec == 2, has its own logic
        else if(adc_bit_prec == 2){
            if ( (pkt >> 27 == 0x1) && (pkt >> 31 == 0x0)){
                unsigned int frm_cnt = (pkt >> 8) & 0x1ff;
                //printf("FRM %u read from DMA\n", frm_cnt);
            } else{
                pkt = pkt >> 2;
                if(pkt >> 29){
                    printf("The Row address packet may be lost !\n");
                    return ERR_ROW_ADDR_LOSS;                    
                }
                else{
                    uint16_t row_pkt_head = pkt >> 22;
                    uint16_t eff_grp_num = (pkt >> 15 ) & 0x7f;
                    uint16_t row_addr = (pkt >> 6) & 0xff;
                    for(int gi = 0; gi < eff_grp_num; gi++){
                        index++;
                        unsigned int grp_pkt = (unsigned int)pvalue[index];
                        packet_tot_size++;
                        grp_pkt = grp_pkt >> 2;
                        if(grp_pkt >> 29){
                            uint16_t g_addr = (grp_pkt >> 24) & 0x1f;
                            unsigned int g_data = grp_pkt & 0xffffff;
                            unsigned int pix_per_grp_last = (g_addr == 0xd) ? 4 : pix_per_grp; // last packet(group addr == 13) only have 4 effecitve data
                            for(uint8_t pi = 0; pi < pix_per_grp_last; pi++){
                                uint16_t pixel_addr = width * row_addr + g_addr * pix_per_grp + pi;
                                int8_t pix_data = (int8_t) ((g_data >> (adc_bit_prec * (pix_per_grp - 1 - pi))) & d_mask);
                                if(pix_data == 1){
                                    pix_data = 127;
                                }else if(pix_data == 3){
                                    pix_data = -127;
                                }
                                if(row_pkt_head == 0x10){ // TempDiff Row address packet flag
                                    temp_diff[pixel_addr] = pix_data;
                                } else if(row_pkt_head == 0x14){ // Spat Diff left Row address packet flag
                                    spat_diff_left[pixel_addr] = pix_data;
                                } else if (row_pkt_head == 0x16){ // Spat Diff right Row address packet flag
                                    spat_diff_right[pixel_addr] = pix_data;
                                } else{
                                    printf("BAD row packt head!\n");
                                }
                            }
                        }
                    }
                }
            }
            index++;
        }
    }
    return packet_tot_size;
}


int rod_decoder_tdsd_size(int *pvalue, int8_t* temp_diff, int8_t* spat_diff_left, int8_t* spat_diff_right, int* td_size, int* sd_size, int height, int width){
    
    //adc_bit_prec in frame head
    unsigned int adc_bit_prec = (unsigned int)pvalue[FRM_HEAD_ADC_PREC_OFFSET] & 0xf;
    unsigned int cnt = (unsigned int)pvalue[FRM_HEAD_FrmCount_OFFSET];
    uint64_t timestamp_high =  (unsigned int)pvalue[FRM_HEAD_TimeStampMSB_OFFSET];
    uint64_t timestamp_low = (unsigned int)pvalue[FRM_HEAD_TimeStampLSB_OFFSET];
    uint64_t timestamp = ((uint64_t)timestamp_high << 32) + (uint64_t)timestamp_low;
   // printf("In fime cnt %d, Timestamp %d\n", cnt, timestamp);
     // set the total pixel number in one group packet
    uint8_t pix_per_grp = 0;
    // set the data mask
    uint16_t d_mask = 0;
    //unsigned int pkt_max_num;
    if(adc_bit_prec == 8){
        pix_per_grp = 2;
        d_mask = 0xff;
     //   pkt_max_num = ROD_8B_ONE_FRM;
    } else if(adc_bit_prec == 4){
        pix_per_grp = 4;
        d_mask = 0xf;
      //  pkt_max_num = ROD_4B_ONE_FRM;
    }
    else if(adc_bit_prec == 2){
        pix_per_grp = 12;
        d_mask = 0x3;
    //    pkt_max_num = ROD_2B_ONE_FRM;
    }else if (adc_bit_prec == 0){
        adc_bit_prec = 8;
      pix_per_grp = 2;
        d_mask = 0xff;      
    } 
    else{
        printf("Wrong ADC precision, %d\n", adc_bit_prec);
       // exit(-1);
        return ERR_WRONG_ADC_PREC;
    }
    //printf("ADC precision %d\n",adc_bit_prec);
    int index = FRM_HEAD_OFFSET;
    //uint16_t width_ui16 = (uint16_t)
    int td_pkt_size = 0;
    int sd_pkt_size = 0;
    int packet_tot_size = 0;
   // for(int ii = 0; ii < pkt_max_num; ii++){
    while(true){
        unsigned int pkt = (unsigned int)pvalue[index];//read a packt date from pvalue
        packet_tot_size++;
        //Detect the end of current frame
        if((pkt == 0xffffffff) || (pkt == 0x0))
        {
            packet_tot_size--;
            break;
        }
            
        // adc bit prec = 8 or 4, share the same logic
        if(adc_bit_prec == 8 || adc_bit_prec == 4){
            pkt = pkt >> 8;
            //bool g_pkt_flag = ;
            if (pkt >> 19 == 0x1){
                unsigned int frm_cnt = pkt & 0x1ff;
                //printf("FRM %u read from DMA\n", frm_cnt);
            }
            else {
                if(pkt >> 23){
                    printf("The Row address packet may be lost !\n");
                    return ERR_ROW_ADDR_LOSS;
                }//else if(pkt == 0xffffffff)
                else {
                    uint16_t row_pkt_head = pkt >> 16;
                    if(row_pkt_head == 0x10){ // TempDiff Row address packet flag
                            td_pkt_size++;
                    } else if(row_pkt_head == 0x14){ // Spat Diff left Row address packet flag
                            sd_pkt_size++;
                    } else if (row_pkt_head == 0x16){ // Spat Diff right Row address packet flag
                        sd_pkt_size++;
                    }
            //if(row_pkt_head == 0x10){
                    uint16_t eff_grp_num = (pkt >> 9 ) & 0x7f;
                    uint16_t row_addr = pkt & 0xff;
                    //Record group data to temp_diff, spat_diff_left and spat_diff_right frame 
                    uint16_t yaddr = width * row_addr;
                    for(int gi = 0; gi < eff_grp_num; gi++){
                        index++;
                        
                        unsigned int grp_pkt = (unsigned int)pvalue[index];

                        packet_tot_size++;
                        if(row_pkt_head == 0x10){ // TempDiff Row address packet flag
                            td_pkt_size++;
                        } else if(row_pkt_head == 0x14){ // Spat Diff left Row address packet flag
                            sd_pkt_size++;
                        } else if (row_pkt_head == 0x16){ // Spat Diff right Row address packet flag
                            sd_pkt_size++;
                        }
                        //if((grp_pkt == 0xffffffff) ||  (grp_pkt == 0x0))
                        //    printf("Wrong group packet, now at last packet");
                        grp_pkt = grp_pkt >> 8;                        
                       /* if((grp_pkt >> 23) == 0){ //group packet flag
                            printf("Wrond group address\n");
                        }*/
                        if(grp_pkt >> 23){
                            uint16_t g_addr = (grp_pkt >> 16) & 0x7f;
                            uint16_t g_data = grp_pkt & 0xffff;
                            //uint16_t xaddr_base = ;
                            //uint16_t pixel_addr = yaddr + g_addr * pix_per_grp;
                            for(uint8_t pi = 0; pi < pix_per_grp; pi++){
                                uint16_t pixel_addr = yaddr + g_addr * pix_per_grp + pi;
                                //pixel_addr += pi;//100us
                                int8_t pix_data = (int8_t) ((g_data >> (adc_bit_prec * (pix_per_grp - 1 - pi))) & d_mask);//100us
                                if(adc_bit_prec == 4){
                                    pix_data = pix_data << 4;
                                }
                                //BYTE pix_data = (BYTE) ((g_data >> ((pix_per_grp - 1 - pi) << 3)) & d_mask);
                                //BYTE pix_data = (BYTE) ((g_data >> (8 * (2 - 1 - pi))) & d_mask);
                                // 230us for save to corresponding buffer
                                if(row_pkt_head == 0x10){ // TempDiff Row address packet flag
                                    temp_diff[pixel_addr] = pix_data;
                                    //td_pkt_size++;
                                } else if(row_pkt_head == 0x14){ // Spat Diff left Row address packet flag
                                    spat_diff_left[pixel_addr] = pix_data;
                                    //sd_pkt_size++;
                                } else if (row_pkt_head == 0x16){ // Spat Diff right Row address packet flag
                                    spat_diff_right[pixel_addr] = pix_data;
                                   // sd_pkt_size++;
                                } else{
                                    printf("BAD row packt head!\n");
                                }/**/
                            } 
                        }
                    }
                }
            }
            index++;
            //packet_tot_size++;
        } 
        // adc bit prec == 2, has its own logic
        else if(adc_bit_prec == 2){
            if ( (pkt >> 27 == 0x1) && (pkt >> 31 == 0x0)){
                unsigned int frm_cnt = (pkt >> 8) & 0x1ff;
                //printf("FRM %u read from DMA\n", frm_cnt);
            } else{
                pkt = pkt >> 2;
                if(pkt >> 29){
                    printf("The Row address packet may be lost !\n");
                    return ERR_ROW_ADDR_LOSS;                    
                }
                else{
                    uint16_t row_pkt_head = pkt >> 22;
                    if(row_pkt_head == 0x10){ // TempDiff Row address packet flag
                        td_pkt_size++;
                    } else if(row_pkt_head == 0x14){ // Spat Diff left Row address packet flag
                        sd_pkt_size++;
                    } else if (row_pkt_head == 0x16){ // Spat Diff right Row address packet flag
                        sd_pkt_size++;
                    }
                    uint16_t eff_grp_num = (pkt >> 15 ) & 0x7f;
                    uint16_t row_addr = (pkt >> 6) & 0xff;
                    for(int gi = 0; gi < eff_grp_num; gi++){
                        index++;
                        unsigned int grp_pkt = (unsigned int)pvalue[index];
                        packet_tot_size++;
                        if(row_pkt_head == 0x10){ // TempDiff Row address packet flag
                            td_pkt_size++;
                        } else if(row_pkt_head == 0x14){ // Spat Diff left Row address packet flag
                            sd_pkt_size++;
                        } else if (row_pkt_head == 0x16){ // Spat Diff right Row address packet flag
                            sd_pkt_size++;
                        }
                        grp_pkt = grp_pkt >> 2;
                        if(grp_pkt >> 29){
                            uint16_t g_addr = (grp_pkt >> 24) & 0x1f;
                            unsigned int g_data = grp_pkt & 0xffffff;
                            unsigned int pix_per_grp_last = (g_addr == 0xd) ? 4 : pix_per_grp; // last packet(group addr == 13) only have 4 effecitve data
                            for(uint8_t pi = 0; pi < pix_per_grp_last; pi++){
                                uint16_t pixel_addr = width * row_addr + g_addr * pix_per_grp + pi;
                                int8_t pix_data = (int8_t) ((g_data >> (adc_bit_prec * (pix_per_grp - 1 - pi))) & d_mask);
                                if(pix_data == 1){
                                    pix_data = 127;
                                }else if(pix_data == 3){
                                    pix_data = -127;
                                }
                                if(row_pkt_head == 0x10){ // TempDiff Row address packet flag
                                    temp_diff[pixel_addr] = pix_data;
                                  //  td_pkt_size++;
                                } else if(row_pkt_head == 0x14){ // Spat Diff left Row address packet flag
                                    spat_diff_left[pixel_addr] = pix_data;
                                  //  sd_pkt_size++;
                                } else if (row_pkt_head == 0x16){ // Spat Diff right Row address packet flag
                                    spat_diff_right[pixel_addr] = pix_data;
                                   // sd_pkt_size++;
                                } else{
                                    printf("BAD row packt head!\n");
                                }
                            }
                        }
                    }
                }
            }
            index++;
        }
    }
    *td_size = td_pkt_size;
    *sd_size = sd_pkt_size;
    //printf("TD size %d, SD size %d\n", td_pkt_size, sd_pkt_size);
    return packet_tot_size;
}


uint64_t rod_decoder_tdsd_size(int *pvalue, int8_t* temp_diff, int8_t* spat_diff_left, int8_t* spat_diff_right, int* td_size, int* sd_size, int height, int width, uint64_t * timestamp, int* fcnt, int* radc_prec){
    
  
   // printf("%x\n", pvalue[0] & 0xff000000 );
    if((pvalue[0] & 0xff000000) == 0xed000000){
     //   printf("New USB header frame head!\n");
        // uint64_t ts_h = static_cast<uint64_t> (pvalue[1] & 0xffffff);
        // uint64_t ts_m = static_cast<uint64_t> (pvalue[2] & 0xffffff);
        // uint64_t ts_l = static_cast<uint64_t> (pvalue[3] & 0xffffff);
        // *timestamp = static_cast<uint64_t> ((ts_h) << 48) + (ts_m << 24) + ts_l;
        // uint32_t cnt_h = static_cast<uint32_t> (pvalue[4] & 0xffffff);
        // uint32_t cnt_l = static_cast<uint32_t> (pvalue[5] & 0xffffff);
        // *fcnt = static_cast<uint32_t> (cnt_h << 24) + cnt_l;
        usb_header_parse(pvalue, timestamp, fcnt, radc_prec);
    } else{
        uint64_t timestamp_high =  (unsigned int)pvalue[FRM_HEAD_TimeStampMSB_OFFSET];
        uint64_t timestamp_low = (unsigned int)pvalue[FRM_HEAD_TimeStampLSB_OFFSET];
        *timestamp = ((uint64_t)timestamp_high << 32) + (uint64_t)timestamp_low;
        *fcnt = pvalue[FRM_HEAD_FrmCount_OFFSET];
        unsigned int adc_bit_prec = (unsigned int)pvalue[FRM_HEAD_ADC_PREC_OFFSET] & 0xf;
        *radc_prec = adc_bit_prec;
    
    }
      //adc_bit_prec in frame head
    unsigned int adc_bit_prec = *radc_prec;
   
    
   // printf("In fime cnt %d, Timestamp %d\n", cnt, timestamp);
     // set the total pixel number in one group packet
    uint8_t pix_per_grp = 0;
    // set the data mask
    uint16_t d_mask = 0;
    //unsigned int pkt_max_num;
    if(adc_bit_prec == 8){
        pix_per_grp = 2;
        d_mask = 0xff;
     //   pkt_max_num = ROD_8B_ONE_FRM;
    } else if(adc_bit_prec == 4){
        pix_per_grp = 4;
        d_mask = 0xf;
      //  pkt_max_num = ROD_4B_ONE_FRM;
    }
    else if(adc_bit_prec == 2){
        pix_per_grp = 12;
        d_mask = 0x3;
    //    pkt_max_num = ROD_2B_ONE_FRM;
    }else if (adc_bit_prec == 0){
        adc_bit_prec = 8;
      pix_per_grp = 2;
        d_mask = 0xff;      
    } 
    else{
        printf("Wrong ADC precision, %d\n", adc_bit_prec);
       // exit(-1);
        return ERR_WRONG_ADC_PREC;
    }
    //printf("ADC precision %d\n",adc_bit_prec);
    int index = FRM_HEAD_OFFSET;
    //uint16_t width_ui16 = (uint16_t)
    int td_pkt_size = 0;
    int sd_pkt_size = 0;
    int packet_tot_size = 0;
   // for(int ii = 0; ii < pkt_max_num; ii++){
    while(true){
        unsigned int pkt = (unsigned int)pvalue[index];//read a packt date from pvalue
        packet_tot_size++;
        //Detect the end of current frame
        if((pkt == 0xffffffff) || (pkt == 0x0))
        {
            packet_tot_size--;
            break;
        }
            
        // adc bit prec = 8 or 4, share the same logic
        if(adc_bit_prec == 8 || adc_bit_prec == 4){
            pkt = pkt >> 8;
            //bool g_pkt_flag = ;
            if (pkt >> 19 == 0x1){
                unsigned int frm_cnt = pkt & 0x1ff;
                //printf("FRM %u read from DMA\n", frm_cnt);
            }
            else {
                if(pkt >> 23){
                    printf("The Row address packet may be lost !\n");
                    return ERR_ROW_ADDR_LOSS;
                }//else if(pkt == 0xffffffff)
                else {
                    uint16_t row_pkt_head = pkt >> 16;
                    if(row_pkt_head == 0x10){ // TempDiff Row address packet flag
                            td_pkt_size++;
                    } else if(row_pkt_head == 0x14){ // Spat Diff left Row address packet flag
                            sd_pkt_size++;
                    } else if (row_pkt_head == 0x16){ // Spat Diff right Row address packet flag
                        sd_pkt_size++;
                    }
            //if(row_pkt_head == 0x10){
                    uint16_t eff_grp_num = (pkt >> 9 ) & 0x7f;
                    uint16_t row_addr = pkt & 0xff;
                    //Record group data to temp_diff, spat_diff_left and spat_diff_right frame 
                    uint16_t yaddr = width * row_addr;
                    for(int gi = 0; gi < eff_grp_num; gi++){
                        index++;
                        
                        unsigned int grp_pkt = (unsigned int)pvalue[index];

                        packet_tot_size++;
                        if(row_pkt_head == 0x10){ // TempDiff Row address packet flag
                            td_pkt_size++;
                        } else if(row_pkt_head == 0x14){ // Spat Diff left Row address packet flag
                            sd_pkt_size++;
                        } else if (row_pkt_head == 0x16){ // Spat Diff right Row address packet flag
                            sd_pkt_size++;
                        }
                        //if((grp_pkt == 0xffffffff) ||  (grp_pkt == 0x0))
                        //    printf("Wrong group packet, now at last packet");
                        grp_pkt = grp_pkt >> 8;                        
                       /* if((grp_pkt >> 23) == 0){ //group packet flag
                            printf("Wrond group address\n");
                        }*/
                        if(grp_pkt >> 23){
                            uint16_t g_addr = (grp_pkt >> 16) & 0x7f;
                            uint16_t g_data = grp_pkt & 0xffff;
                            //uint16_t xaddr_base = ;
                            //uint16_t pixel_addr = yaddr + g_addr * pix_per_grp;
                            for(uint8_t pi = 0; pi < pix_per_grp; pi++){
                                uint16_t pixel_addr = yaddr + g_addr * pix_per_grp + pi;
                                //pixel_addr += pi;//100us
                                int8_t pix_data = (int8_t) ((g_data >> (adc_bit_prec * (pix_per_grp - 1 - pi))) & d_mask);//100us
                                if(adc_bit_prec == 4){
                                    pix_data = pix_data << 4;
                                }
                                //BYTE pix_data = (BYTE) ((g_data >> ((pix_per_grp - 1 - pi) << 3)) & d_mask);
                                //BYTE pix_data = (BYTE) ((g_data >> (8 * (2 - 1 - pi))) & d_mask);
                                // 230us for save to corresponding buffer
                                if(row_pkt_head == 0x10){ // TempDiff Row address packet flag
                                    temp_diff[pixel_addr] = pix_data;
                                    //td_pkt_size++;
                                } else if(row_pkt_head == 0x14){ // Spat Diff left Row address packet flag
                                    spat_diff_left[pixel_addr] = pix_data;
                                    //sd_pkt_size++;
                                } else if (row_pkt_head == 0x16){ // Spat Diff right Row address packet flag
                                    spat_diff_right[pixel_addr] = pix_data;
                                   // sd_pkt_size++;
                                } else{
                                    printf("BAD row packt head!\n");
                                }/**/
                            } 
                        }
                    }
                }
            }
            index++;
            //packet_tot_size++;
        } 
        // adc bit prec == 2, has its own logic
        else if(adc_bit_prec == 2){
            if ( (pkt >> 27 == 0x1) && (pkt >> 31 == 0x0)){
                unsigned int frm_cnt = (pkt >> 8) & 0x1ff;
                //printf("FRM %u read from DMA\n", frm_cnt);
            } else{
                pkt = pkt >> 2;
                if(pkt >> 29){
                    printf("The Row address packet may be lost !\n");
                    return ERR_ROW_ADDR_LOSS;                    
                }
                else{
                    uint16_t row_pkt_head = pkt >> 22;
                    if(row_pkt_head == 0x10){ // TempDiff Row address packet flag
                        td_pkt_size++;
                    } else if(row_pkt_head == 0x14){ // Spat Diff left Row address packet flag
                        sd_pkt_size++;
                    } else if (row_pkt_head == 0x16){ // Spat Diff right Row address packet flag
                        sd_pkt_size++;
                    }
                    uint16_t eff_grp_num = (pkt >> 15 ) & 0x7f;
                    uint16_t row_addr = (pkt >> 6) & 0xff;
                    for(int gi = 0; gi < eff_grp_num; gi++){
                        index++;
                        unsigned int grp_pkt = (unsigned int)pvalue[index];
                        packet_tot_size++;
                        if(row_pkt_head == 0x10){ // TempDiff Row address packet flag
                            td_pkt_size++;
                        } else if(row_pkt_head == 0x14){ // Spat Diff left Row address packet flag
                            sd_pkt_size++;
                        } else if (row_pkt_head == 0x16){ // Spat Diff right Row address packet flag
                            sd_pkt_size++;
                        }
                        grp_pkt = grp_pkt >> 2;
                        if(grp_pkt >> 29){
                            uint16_t g_addr = (grp_pkt >> 24) & 0x1f;
                            unsigned int g_data = grp_pkt & 0xffffff;
                            unsigned int pix_per_grp_last = (g_addr == 0xd) ? 4 : pix_per_grp; // last packet(group addr == 13) only have 4 effecitve data
                            for(uint8_t pi = 0; pi < pix_per_grp_last; pi++){
                                uint16_t pixel_addr = width * row_addr + g_addr * pix_per_grp + pi;
                                int8_t pix_data = (int8_t) ((g_data >> (adc_bit_prec * (pix_per_grp - 1 - pi))) & d_mask);
                                if(pix_data == 1){
                                    pix_data = 127;
                                }else if(pix_data == 3){
                                    pix_data = -127;
                                }
                                if(row_pkt_head == 0x10){ // TempDiff Row address packet flag
                                    temp_diff[pixel_addr] = pix_data;
                                  //  td_pkt_size++;
                                } else if(row_pkt_head == 0x14){ // Spat Diff left Row address packet flag
                                    spat_diff_left[pixel_addr] = pix_data;
                                  //  sd_pkt_size++;
                                } else if (row_pkt_head == 0x16){ // Spat Diff right Row address packet flag
                                    spat_diff_right[pixel_addr] = pix_data;
                                   // sd_pkt_size++;
                                } else{
                                    printf("BAD row packt head!\n");
                                }
                            }
                        }
                    }
                }
            }
            index++;
        }
    }
    *td_size = td_pkt_size;
    *sd_size = sd_pkt_size;
    //printf("TD size %d, SD size %d\n", td_pkt_size, sd_pkt_size);
    return packet_tot_size;
}

int cone_reader(int *pvalue, int16_t* raw, int height, int width){
    //timestamp for sync CONE and ROD
    //uint64_t timestamp_high =  (unsigned int)pvalue[CONE_FRM_HEAD_TimeStampMSB_OFFSET];
    //uint64_t timestamp_low = (unsigned int)pvalue[CONE_FRM_HEAD_TimeStampLSB_OFFSET];
   // uint64_t timestamp = (timestamp_high << 32) + timestamp_low;

    int index = 16;
    for(int ii = 0; ii < height * width; ii++){
        int16_t pix = (int16_t) ((unsigned int)pvalue[index] & 0x3ff);
        //unsigned int pixel_addr = width * row_addr + g_addr * pix_per_grp + pi;
        raw[ii] = pix;
        index++;
    }
    return 0;
}
int cone_reader(int *pvalue, int16_t* raw, int height, int width, uint64_t* timestamp, int* fcnt){
    //timestamp for sync CONE and ROD
    uint64_t timestamp_high =  (unsigned int)pvalue[CONE_FRM_HEAD_TimeStampMSB_OFFSET];
    uint64_t timestamp_low = (unsigned int)pvalue[CONE_FRM_HEAD_TimeStampLSB_OFFSET];
    *timestamp = ((uint64_t)timestamp_high << 32) + (uint64_t)timestamp_low;
    
    *fcnt = pvalue[CONE_FRM_HEAD_FrmCount_OFFSET];
    
    int index = 16;
    for(int ii = 0; ii < height * width; ii++){
        int16_t pix = (int16_t) ((unsigned int)pvalue[index] & 0x3ff);
        //unsigned int pixel_addr = width * row_addr + g_addr * pix_per_grp + pi;
        raw[ii] = pix;
        index++;
    }
    return 0;
}
int cone_reader_usb_info(int *pvalue, int16_t* raw, int height, int width, uint64_t* timestamp, int* fcnt){

    // uint64_t ts_h = static_cast<uint64_t> (pvalue[1] & 0xffffff);
    // uint64_t ts_m = static_cast<uint64_t> (pvalue[2] & 0xffffff);
    // uint64_t ts_l = static_cast<uint64_t> (pvalue[3] & 0xffffff);
    // *timestamp = static_cast<uint64_t> ((ts_h) << 48) + (ts_m << 24) + ts_l;
    // uint32_t cnt_h = static_cast<uint32_t> (pvalue[4] & 0xffffff);
    // uint32_t cnt_l = static_cast<uint32_t> (pvalue[5] & 0xffffff);
    // *fcnt = static_cast<uint32_t> (cnt_h << 24) + cnt_l;
    int radc_prec;
    usb_header_parse(pvalue, timestamp, fcnt, &radc_prec);
    int index = 16;
    int* pvalue_nohead = &pvalue[index];
   // printf("Cone cnt %d, TS %lld\n", *fcnt, *timestamp);
    //cv::Mat pval_mat(height, width, CV_32SC1, pvalue_nohead);
    for(int i = 0; i < height * width /2 ; i++){
        int two_pix = pvalue_nohead[i];
        int16_t pix = two_pix & 0x3ff;
        int16_t pix2 = (two_pix >> 10) & 0x3ff;

        //unsigned int pixel_addr = width * row_addr + g_addr * pix_per_grp + pi;
        raw[2 * i] = pix;
        raw[2 * i + 1] = pix2;
    }
    /*for (auto y = 0; y < height; y++) {
        //auto raw_ptr = raw.ptr<int16_t>(y);
    // auto pval_mat_ptr = pval_mat.ptr<int>(y);
        for (auto x = 0; x < width; x += 2) {
            int pos = y * int(width/2) + int(x / 2);
            //printf("y %d, x %d, pos %d \n", y, x, pos);
            int two_pix = pvalue_nohead[pos];
            int16_t pix = two_pix & 0x3ff;
            int16_t pix2 = (two_pix >> 10) & 0x3ff;
            if (y % 4 == 0 || y % 4 == 1) {
                raw[y * width + 2 * x + 2] = pix;
                raw[y * width + 2 * x + 3] = pix2;
            }
            else if (y % 4 == 2 || y % 4 == 3) {
                raw[y * width + 2 * x] = pix;
                raw[y * width + 2 * x + 1] = pix2;
            }
        }

    }*/
    return 0;

}

int cone_reader_usb(int *pvalue, int16_t* raw, int height, int width){

     int index = 16;
    int* pvalue_nohead = &pvalue[index];
    //cv::Mat pval_mat(height, width, CV_32SC1, pvalue_nohead);
    for(int i = 0; i < height * width /2 ; i++){
        int two_pix = pvalue_nohead[i];
        int16_t pix = two_pix & 0x3ff;
        int16_t pix2 = (two_pix >> 10) & 0x3ff;
        raw[2 * i] = pix;
        raw[2 * i + 1] = pix;
    }
    return 0;

}
// ----------------
// Python interface
// ----------------

namespace py = pybind11;
// wrap C++ function with NumPy array IO
/*py::array_t<int> rod_decoder_py(py::array_t<int, py::array::c_style | py::array::forcecast> pvalue_np, py::array_t<int8_t, py::array::c_style | py::array::forcecast> temp_diff_np,  py::array::c_style | py::array::forcecast> spat_diff_left_np, py::array::c_style | py::array::forcecast> spat_diff_right_np, int height, int width)*/
int rod_decoder_py(py::array_t<int>& pvalue_np, py::array_t<int8_t>& temp_diff_np, py::array_t<int8_t>& spat_diff_left_np, py::array_t<int8_t>& spat_diff_right_np, int height, int width)
{

    py::buffer_info pvalue_buf = pvalue_np.request();// 获取arr1, arr2的信息
    py::buffer_info td_buf = temp_diff_np.request();
    py::buffer_info sd_l_buf = spat_diff_left_np.request();
    py::buffer_info sd_r_buf = spat_diff_right_np.request();

    //int* pvalue = (int*) malloc(pvalue_np.size());
    int* pval_ptr = (int *) pvalue_buf.ptr;
    int8_t* td_ptr = (int8_t*) td_buf.ptr;
    int8_t* sd_l_ptr = (int8_t*) sd_l_buf.ptr;
    int8_t* sd_r_ptr = (int8_t*) sd_r_buf.ptr;

    int ret = rod_decoder(pval_ptr, td_ptr, sd_l_ptr, sd_r_ptr, height, width);
    return ret;
}

int rod_decoder_py_bulk(py::bytes& pvalue, py::array_t<int8_t>& temp_diff_np, py::array_t<int8_t>& spat_diff_left_np, py::array_t<int8_t>& spat_diff_right_np, int height, int width)
{

    //py::buffer_info pvalue_buf = pvalue_np.request();// 获取arr1, arr2的信息
    py::buffer_info td_buf = temp_diff_np.request();
    py::buffer_info sd_l_buf = spat_diff_left_np.request();
    py::buffer_info sd_r_buf = spat_diff_right_np.request();

    //int* pvalue = (int*) malloc(pvalue_np.size());
    //Py_ssize_t size = PyBytes_GET_SIZE(pvalue.ptr());
    int* pval_ptr = (int*) pvalue.ptr();
   // int* pval_ptr = (int *) pvalue;
    int8_t* td_ptr = (int8_t*) td_buf.ptr;
    int8_t* sd_l_ptr = (int8_t*) sd_l_buf.ptr;
    int8_t* sd_r_ptr = (int8_t*) sd_r_buf.ptr;

    int ret = rod_decoder(pval_ptr, td_ptr, sd_l_ptr, sd_r_ptr, height, width);
    return ret;
}

int rod_decoder_py_byfile(const std::string &fpath, int img_per_file, int size, int one_frm_size, py::array_t<int8_t>& temp_diff_np, py::array_t<int8_t>& spat_diff_left_np, py::array_t<int8_t>& spat_diff_right_np, int height, int width)
{
    //std::cout<<"The file path is "<<fpath<<"read size"<<size<<"img_per_file "<<img_per_file <<"one_frm_size "<<one_frm_size <<std::endl;

    std::ifstream fin(fpath, std::ios::in | std::ios::binary);
    int *pvalue = (int *)calloc(size , sizeof(int));
    fin.read((char *) pvalue, size * sizeof(int));
    fin.close();
    //py::buffer_info pvalue_buf = pvalue_np.request();// 获取arr1, arr2的信息
    py::buffer_info td_buf = temp_diff_np.request();
    py::buffer_info sd_l_buf = spat_diff_left_np.request();
    py::buffer_info sd_r_buf = spat_diff_right_np.request();


   int diffoffset = width * height;
   int f_offset = one_frm_size;
    int8_t* td_ptr = (int8_t*) td_buf.ptr;
    int8_t* sd_l_ptr = (int8_t*) sd_l_buf.ptr;
    int8_t* sd_r_ptr = (int8_t*) sd_r_buf.ptr;
    int ret = 0;
   for(int i = 0; i < img_per_file; i++){
      //  std::cout<<"processing "<<i<<" th img"<<std::endl;
        int* pval_ptr = pvalue + f_offset * i;
        int8_t* td_val = td_ptr + diffoffset * i;
        int8_t* sd_l_val = sd_l_ptr + diffoffset * i;
        int8_t* sd_r_val = sd_r_ptr + diffoffset * i;
        ret = rod_decoder(pval_ptr, td_val, sd_l_val, sd_r_val, height, width);
   }

   free(pvalue);
    return ret;
}
// with bandwith calculation
int rod_decoder_py_byfile_bw(const std::string &fpath, int img_per_file, int size, int one_frm_size, 
                            py::array_t<int8_t>& temp_diff_np, py::array_t<int8_t>& spat_diff_left_np, py::array_t<int8_t>& spat_diff_right_np, 
                            py::array_t<int>& diff_size,
                            int height, int width)
{
    //std::cout<<"The file path is "<<fpath<<"read size"<<size<<"img_per_file "<<img_per_file <<"one_frm_size "<<one_frm_size <<std::endl;

    std::ifstream fin(fpath, std::ios::in | std::ios::binary);
    int *pvalue = (int *)calloc(size , sizeof(int));
    fin.read((char *) pvalue, size * sizeof(int));
    fin.close();
    //py::buffer_info pvalue_buf = pvalue_np.request();// 获取arr1, arr2的信息
    py::buffer_info td_buf = temp_diff_np.request();
    py::buffer_info sd_l_buf = spat_diff_left_np.request();
    py::buffer_info sd_r_buf = spat_diff_right_np.request();
    py::buffer_info ds_buf = diff_size.request();

    //int* pvalue = (int*) malloc(pvalue_np.size());
    //Py_ssize_t size = PyBytes_GET_SIZE(pvalue.ptr());
    //int* pval_ptr = (int*) pvalue.ptr();
   // int* pval_ptr = (int *) pvalue;
   int diffoffset = width * height;
   int f_offset = one_frm_size;
    int8_t* td_ptr = (int8_t*) td_buf.ptr;
    int8_t* sd_l_ptr = (int8_t*) sd_l_buf.ptr;
    int8_t* sd_r_ptr = (int8_t*) sd_r_buf.ptr;
    int* ds_ptr = (int*) ds_buf.ptr;
    int ret = 0;
   for(int i = 0; i < img_per_file; i++){
      //  std::cout<<"processing "<<i<<" th img"<<std::endl;
        int* pval_ptr = pvalue + f_offset * i;
        int8_t* td_val = td_ptr + diffoffset * i;
        int8_t* sd_l_val = sd_l_ptr + diffoffset * i;
        int8_t* sd_r_val = sd_r_ptr + diffoffset * i;
        ret = rod_decoder(pval_ptr, td_val, sd_l_val, sd_r_val, height, width);
        int* ds_val = ds_ptr + i;
        *ds_val = ret;
   }


   free(pvalue);
    return ret;
}

int rod_compact_pkt(const std::string &fpath, int img_per_file, int size, int one_frm_size, 
                py::array_t<int32_t>& rod_pkt_np)
{
    //std::cout<<"The file path is "<<fpath<<"read size"<<size<<"img_per_file "<<img_per_file <<"one_frm_size "<<one_frm_size <<std::endl;

    std::ifstream fin(fpath, std::ios::in | std::ios::binary);
    int *pvalue = (int *)calloc(size , sizeof(int));
    fin.read((char *) pvalue, size * sizeof(int));
    fin.close();
     //int *rod_pkt = (int *)calloc(size , sizeof(int));//compact_pkt
    //py::buffer_info pvalue_buf = pvalue_np.request();// 获取arr1, arr2的信息
    py::buffer_info pkt_buf = rod_pkt_np.request();
   
   // int diffoffset = width * height;
    int f_offset = one_frm_size;
    int32_t* pkt_buf_ptr = (int32_t*) pkt_buf.ptr;

    //int ret;
    
    int total_pkt_num = 0;
    int pktptr_this_file = 0;
    for(int i = 0; i < img_per_file; i++){
        //  std::cout<<"processing "<<i<<" th img"<<std::endl;
            int* pval_ptr = pvalue + f_offset * i;
            int index = 0;
            //int pktptr_this_frm = 0;
            while(true){
              //  unsigned int pkt = (unsigned int) pval_ptr[index];
                if((unsigned int) pval_ptr[index] !=  0xffffffff){//end of one frame packet 
                    pkt_buf_ptr[pktptr_this_file] = pvalue[index];
                    pktptr_this_file++;
                    index++;
                }else{
                    pkt_buf_ptr[pktptr_this_file - index] |=  (index << 4) | 0x80000000; // packet num in this frame
                    printf("packt num in %d frame: %d, pkt_header 0x%x\n", i, index, pkt_buf_ptr[pktptr_this_file - index]);
                    total_pkt_num += index;
                    break;
                }
            }
        // int8_t* td_val = td_ptr + diffoffset * i;
    }


   free(pvalue);
    return total_pkt_num;
}


int rod_decoder_py_byfile_td_sd_bw(const std::string &fpath, int img_per_file, int size, int one_frm_size, 
                            py::array_t<int8_t>& temp_diff_np, py::array_t<int8_t>& spat_diff_left_np, py::array_t<int8_t>& spat_diff_right_np, 
                            py::array_t<int>& diff_size, py::array_t<int>& td_size, py::array_t<int>& sd_size,
                            int height, int width)
{
    //std::cout<<"The file path is "<<fpath<<"read size"<<size<<"img_per_file "<<img_per_file <<"one_frm_size "<<one_frm_size <<std::endl;

    std::ifstream fin(fpath, std::ios::in | std::ios::binary);
    int *pvalue = (int *)calloc(size , sizeof(int));
    fin.read((char *) pvalue, size * sizeof(int));
    fin.close();
    //py::buffer_info pvalue_buf = pvalue_np.request();// 获取arr1, arr2的信息
    py::buffer_info td_buf = temp_diff_np.request();
    py::buffer_info sd_l_buf = spat_diff_left_np.request();
    py::buffer_info sd_r_buf = spat_diff_right_np.request();
    py::buffer_info ds_buf = diff_size.request();
    py::buffer_info tds_buf = td_size.request();
    py::buffer_info sds_buf = sd_size.request();

    //int* pvalue = (int*) malloc(pvalue_np.size());
    //Py_ssize_t size = PyBytes_GET_SIZE(pvalue.ptr());
    //int* pval_ptr = (int*) pvalue.ptr();
   // int* pval_ptr = (int *) pvalue;
    int diffoffset = width * height;
    int f_offset = one_frm_size;
    int8_t* td_ptr = (int8_t*) td_buf.ptr;
    int8_t* sd_l_ptr = (int8_t*) sd_l_buf.ptr;
    int8_t* sd_r_ptr = (int8_t*) sd_r_buf.ptr;
    int* ds_ptr = (int*) ds_buf.ptr;
     int* tds_ptr = (int*) tds_buf.ptr;
     int* sds_ptr = (int*) sds_buf.ptr;
    int ret  = 0;
    for(int i = 0; i < img_per_file; i++){
        //  std::cout<<"processing "<<i<<" th img"<<std::endl;
            int* pval_ptr = pvalue + f_offset * i;
            int8_t* td_val = td_ptr + diffoffset * i;
            int8_t* sd_l_val = sd_l_ptr + diffoffset * i;
            int8_t* sd_r_val = sd_r_ptr + diffoffset * i;
            int td_size_val, sd_size_val;
            ret = rod_decoder_tdsd_size(pval_ptr, td_val, sd_l_val, sd_r_val, &td_size_val, &sd_size_val, height, width);
            int* ds_val = ds_ptr + i;
            *ds_val = ret;
            int* tds_val = tds_ptr + i;
            *tds_val = td_size_val;
            int* sds_val = sds_ptr + i;
            *sds_val = sd_size_val;
    }


   free(pvalue);
    return ret;
}

int cone_reader_py_byfile(const std::string &fpath, int size, py::array_t<int16_t>& raw_np, int height, int width)
{
   // std::cout<<"The file path is "<<fpath<<"read size"<<size <<std::endl;

    std::ifstream fin(fpath, std::ios::in | std::ios::binary);
    int *pvalue = (int *)calloc(size , sizeof(int));
    fin.read((char *) pvalue, size * sizeof(int));
    fin.close();
    //py::buffer_info pvalue_buf = pvalue_np.request();// 获取arr1, arr2的信息
    py::buffer_info raw_buf = raw_np.request();


    //int* pvalue = (int*) malloc(pvalue_np.size());
    //Py_ssize_t size = PyBytes_GET_SIZE(pvalue.ptr());
    //int* pval_ptr = (int*) pvalue.ptr();
   // int* pval_ptr = (int *) pvalue;
    int16_t* raw_ptr = (int16_t*) raw_buf.ptr;
    int ret;
    ret = cone_reader(pvalue, raw_ptr, height, width);
   free(pvalue);

    return ret;
}

int cone_reader_py_fullInfo(const std::string &fpath, int size, py::array_t<int16_t>& raw_np,
                        py::array_t<uint64_t>& timestamp, py::array_t<int>& fcount, py::array_t<int>& adc_bit_prec,
                        int height, int width)
{
   // std::cout<<"The file path is "<<fpath<<"read size"<<size <<std::endl;

    std::ifstream fin(fpath, std::ios::in | std::ios::binary);
    int *pvalue = (int *)calloc(size , sizeof(int));
    fin.read((char *) pvalue, size * sizeof(int));
    fin.close();
    //py::buffer_info pvalue_buf = pvalue_np.request();// 获取arr1, arr2的信息
    py::buffer_info raw_buf = raw_np.request();
    py::buffer_info timestamp_buf = timestamp.request();
    py::buffer_info fcount_buf = fcount.request();
    py::buffer_info adc_bit_prec_buf = adc_bit_prec.request();

    uint64_t* timestamp_ptr = (uint64_t*) timestamp_buf.ptr;
    int* fcnt_ptr = (int*) fcount_buf.ptr;
    int* adcprec_ptr = (int*) adc_bit_prec_buf.ptr;
    //int* pvalue = (int*) malloc(pvalue_np.size());
    //Py_ssize_t size = PyBytes_GET_SIZE(pvalue.ptr());
    //int* pval_ptr = (int*) pvalue.ptr();
   // int* pval_ptr = (int *) pvalue;
    int16_t* raw_ptr = (int16_t*) raw_buf.ptr;
    int ret;
    int fcnt;
    uint64_t ctimestamp;
    ret = cone_reader(pvalue, raw_ptr, height, width, &ctimestamp, &fcnt);
    
    *timestamp_ptr = ctimestamp;
    *fcnt_ptr = fcnt;
    *adcprec_ptr = 10;
   // printf("Test cone timestamp %lu cnt %i\n", ctimestamp, fcnt);
    free(pvalue);

    return ret;
}

int rod_decoder_py_fullInfo(const std::string &fpath, int img_per_file, int size, int one_frm_size, 
                            py::array_t<int8_t>& temp_diff_np, py::array_t<int8_t>& spat_diff_left_np, py::array_t<int8_t>& spat_diff_right_np, 
                            py::array_t<int>& diff_size, py::array_t<int>& td_size, py::array_t<int>& sd_size,
                            py::array_t<uint64_t>& timestamp, py::array_t<int>& fcnt, py::array_t<int>& adc_bit_prec,
                            int height, int width)
{
    //std::cout<<"The file path is "<<fpath<<"read size"<<size<<"img_per_file "<<img_per_file <<"one_frm_size "<<one_frm_size <<std::endl;

    std::ifstream fin(fpath, std::ios::in | std::ios::binary);
    int *pvalue = (int *)calloc(size , sizeof(int));
    fin.read((char *) pvalue, size * sizeof(int));
    fin.close();
    //py::buffer_info pvalue_buf = pvalue_np.request();// 获取arr1, arr2的信息
    py::buffer_info td_buf = temp_diff_np.request();
    py::buffer_info sd_l_buf = spat_diff_left_np.request();
    py::buffer_info sd_r_buf = spat_diff_right_np.request();
    py::buffer_info ds_buf = diff_size.request();
    py::buffer_info tds_buf = td_size.request();
    py::buffer_info sds_buf = sd_size.request();
    py::buffer_info timestamp_buf = timestamp.request();
    py::buffer_info fcount_buf = fcnt.request();
    py::buffer_info adc_bit_prec_buf = adc_bit_prec.request();

    uint64_t* timestamp_ptr = (uint64_t*) timestamp_buf.ptr;
    int* fcnt_ptr = (int*) fcount_buf.ptr;
    int* adcprec_ptr = (int*) adc_bit_prec_buf.ptr;
    //int* pvalue = (int*) malloc(pvalue_np.size());
    //Py_ssize_t size = PyBytes_GET_SIZE(pvalue.ptr());
    //int* pval_ptr = (int*) pvalue.ptr();
   // int* pval_ptr = (int *) pvalue;
    int diffoffset = width * height;
    int f_offset = one_frm_size;
    int8_t* td_ptr = (int8_t*) td_buf.ptr;
    int8_t* sd_l_ptr = (int8_t*) sd_l_buf.ptr;
    int8_t* sd_r_ptr = (int8_t*) sd_r_buf.ptr;
    int* ds_ptr = (int*) ds_buf.ptr;
     int* tds_ptr = (int*) tds_buf.ptr;
     int* sds_ptr = (int*) sds_buf.ptr;
    int ret  = 0;
    for(int i = 0; i < img_per_file; i++){
        //  std::cout<<"processing "<<i<<" th img"<<std::endl;
            int* pval_ptr = pvalue + f_offset * i;
            int8_t* td_val = td_ptr + diffoffset * i;
            int8_t* sd_l_val = sd_l_ptr + diffoffset * i;
            int8_t* sd_r_val = sd_r_ptr + diffoffset * i;
            int td_size_val, sd_size_val;
            int radc_prec, rfcnt;
            uint64_t rtimestamp;
            ret = rod_decoder_tdsd_size(pval_ptr, td_val, sd_l_val, sd_r_val, &td_size_val, &sd_size_val, height, width, &rtimestamp, &rfcnt, &radc_prec);
            int* ds_val = ds_ptr + i;
            *ds_val = ret;
            int* tds_val = tds_ptr + i;
            *tds_val = td_size_val;
            int* sds_val = sds_ptr + i;
            *sds_val = sd_size_val;
            uint64_t* ts_val = timestamp_ptr + i;
            *ts_val = rtimestamp;
            int* cnt_val = fcnt_ptr + i;
            *cnt_val = rfcnt;
            int* adc_prec_val = adcprec_ptr + i;
            *adc_prec_val = radc_prec;           
    }


   free(pvalue);
    return ret;
}

int rod_decoder_py_onlyinfo(const std::string &fpath, int img_per_file, int size, int one_frm_size, 
                            py::array_t<uint64_t>& timestamp, py::array_t<int>& fcnt, py::array_t<int>& adc_bit_prec,
                            int height, int width){
    std::ifstream fin(fpath, std::ios::in | std::ios::binary);
    int *pvalue = (int *)calloc(size , sizeof(int));
    fin.read((char *) pvalue, size * sizeof(int));
    fin.close();    
    
    py::buffer_info timestamp_buf = timestamp.request();
    py::buffer_info fcount_buf = fcnt.request();
    py::buffer_info adc_bit_prec_buf = adc_bit_prec.request();

    uint64_t* timestamp_ptr = (uint64_t*) timestamp_buf.ptr;
    int* fcnt_ptr = (int*) fcount_buf.ptr;
    int* adcprec_ptr = (int*) adc_bit_prec_buf.ptr;         

    int f_offset = one_frm_size;                    
    for(int i = 0; i < img_per_file; i++){
        int* pval_ptr = pvalue + f_offset * i;
        int radc_prec = pval_ptr[FRM_HEAD_ADC_PREC_OFFSET] & 0xf;
        int rfcnt = pval_ptr[FRM_HEAD_FrmCount_OFFSET];
        uint64_t timestamp_high =  (unsigned int)pval_ptr[FRM_HEAD_TimeStampMSB_OFFSET];
        uint64_t timestamp_low = (unsigned int)pval_ptr[FRM_HEAD_TimeStampLSB_OFFSET];
        uint64_t rtimestamp = ((uint64_t)timestamp_high << 32) + (uint64_t)timestamp_low;
        uint64_t* ts_val = timestamp_ptr + i;
        *ts_val = rtimestamp;
        int* cnt_val = fcnt_ptr + i;
        *cnt_val = rfcnt;
        int* adc_prec_val = adcprec_ptr + i;
        *adc_prec_val = radc_prec;    
     }
     return 0;
}


std::vector<uint64_t> usbfmt_contruct_frm_list(const std::string &fpath, py::list& timestamp_list, py::list& fcnt_list){//
    std::ifstream fin(fpath, std::ios::in | std::ios::binary);
    std::vector<uint64_t> frm_start_Ptrlist;
    if(!fin){
        printf("Can not open file %s\n", fpath.c_str());
        return frm_start_Ptrlist;
    }
    uint64_t file_size = std::filesystem::file_size(fpath);
    uint64_t readstart_ptr = 0;
    uint64_t readend_ptr = 0;
    uint64_t frame_size = 0;
    int frm_num = 0;
    int *pval_head = (int*) malloc(16 * sizeof(int));
    printf("Construct begin %s, file size %ld\n", fpath.c_str(), file_size);
    int fnum = 0;
    while(readstart_ptr < file_size){
        //printf("readstart_ptr %d\n", readstart_ptr);
        //fin.seekg(readstart_ptr + 6 * 4);
        fin.seekg(readstart_ptr);
        fnum++;
        fin.read((char *)pval_head, 16 * sizeof(int));
        uint64_t timestamp;
        int fcnt;
        int radc_prec;
        usb_header_parse(pval_head, &timestamp, &fcnt, &radc_prec);
        frame_size = pval_head[6] & 0xffffff;
        //printf("%lx,%x,%x,%x, %d\n ", timestamp, fcnt, radc_prec, frame_size, fnum);
        timestamp_list.append(timestamp);
        fcnt_list.append(fcnt);
        frm_start_Ptrlist.push_back(readstart_ptr);
        readstart_ptr = frame_size * 4 + readstart_ptr;
        frm_num ++;
    }
    fin.close();
    free(pval_head);
    return frm_start_Ptrlist;
}

// py::generator usbfmt_get_multiple_rod(const std::string &fpath, int frm_start_pos, int read_num, int width, int height){
//     std::ifstream fin(fpath, std::ios::in | std::ios::binary);
//     if(!fin){
//         printf("Can not open file %s\n", fpath.c_str());
//         return -1;
//     }
//     int file_size = getFileSize(fpath.c_str());//Byte109110
//     int frame_size = 0;
//     int reader_frm_start_pos = frm_start_pos;
//     for(auto i = 0; i < read_num; i++){
//         if(reader_frm_start_pos >= file_size){
//             break;
//         }
//         fin.seekg(reader_frm_start_pos + 6 * 4);
//         fin.read((char *) &frame_size, 4);
//         fin.seekg(reader_frm_start_pos);
//         int *pvalue = (int *)calloc(frame_size , sizeof(int));
//         fin.read((char *) pvalue, frame_size * sizeof(int));
//         //py::array_t<char> array({static_cast<py::ssize_t>(bytesRead)}, {1}, buffer.data());
//         reader_frm_start_pos = frame_size * 4 + reader_frm_start_pos;
//         std::array<int8_t, width * height> temp_diff;
//         temp_diff.fill(0);
//         std::array<int8_t, width * height> spat_diff_left;
//         spat_diff_left.fill(0);
//         std::array<int8_t, width * height> spat_diff_right;
//         spat_diff_right.fill(0);
//         int td_size_val, sd_size_val;
//         uint64_t rtimestamp;
//         int radc_prec, rfcnt;
//         // decoder
//         int ret = rod_decoder_tdsd_size(pvalue, temp_diff.data(), spat_diff_left.data(), spat_diff_right.data(), &td_size_val, &sd_size_val, height, width, &rtimestamp, &rfcnt, &radc_prec);
//         std::array td_sd_comp = temp_diff; 
//         td_sd_comp.insert(td_sd_comp.end(), spat_diff_left.begin(), spat_diff_left.end()); 
//         td_sd_comp.insert(td_sd_comp.end(), spat_diff_right.begin(), spat_diff_right.end());
//         free(pvalue);
//     }
//     fin.close();
// }


int usbfmt_get_one_rod_fullinfo(const std::string &fpath, uint64_t frm_start_pos,
                        py::array_t<int8_t>& temp_diff_np, py::array_t<int8_t>& spat_diff_left_np, py::array_t<int8_t>& spat_diff_right_np, 
                        py::array_t<int>& diff_size, py::array_t<int>& td_size, py::array_t<int>& sd_size,
                        py::array_t<uint64_t>& timestamp, py::array_t<int>& fcnt, py::array_t<int>& adc_bit_prec,
                        int height, int width){

    
    std::ifstream fin(fpath, std::ios::in | std::ios::binary);
    if(!fin){
        printf("Can not open file %s\n", fpath.c_str());
        return -1;
    }
    int file_size = std::filesystem::file_size(fpath);
    int frame_size = 0;
    fin.seekg(frm_start_pos + 6 * 4);
    fin.read((char *) &frame_size, 4);
    fin.seekg(frm_start_pos);
    frame_size = frame_size & 0xffffff;
    int *pvalue = (int *)calloc(frame_size , sizeof(int));
    fin.read((char *) pvalue, frame_size * sizeof(int));
    fin.close();
    py::buffer_info td_buf = temp_diff_np.request();
    py::buffer_info sd_l_buf = spat_diff_left_np.request();
    py::buffer_info sd_r_buf = spat_diff_right_np.request();
    py::buffer_info ds_buf = diff_size.request();
    py::buffer_info tds_buf = td_size.request();
    py::buffer_info sds_buf = sd_size.request();
    py::buffer_info timestamp_buf = timestamp.request();
    py::buffer_info fcount_buf = fcnt.request();
    py::buffer_info adc_bit_prec_buf = adc_bit_prec.request();

    uint64_t* timestamp_ptr = (uint64_t*) timestamp_buf.ptr;
    int* fcnt_ptr = (int*) fcount_buf.ptr;
    int* adcprec_ptr = (int*) adc_bit_prec_buf.ptr;

    int diffoffset = width * height;
    //int f_offset = one_frm_size;
    int8_t* td_ptr = (int8_t*) td_buf.ptr;
    int8_t* sd_l_ptr = (int8_t*) sd_l_buf.ptr;
    int8_t* sd_r_ptr = (int8_t*) sd_r_buf.ptr;
    int* ds_ptr = (int*) ds_buf.ptr;
     int* tds_ptr = (int*) tds_buf.ptr;
     int* sds_ptr = (int*) sds_buf.ptr;

    int td_size_val, sd_size_val;
    int radc_prec, rfcnt;
    uint64_t rtimestamp;
    //int* pval_ptr = pvalue;
   int  ret = rod_decoder_tdsd_size(pvalue, td_ptr, sd_l_ptr, sd_r_ptr, &td_size_val, &sd_size_val, height, width, &rtimestamp, &rfcnt, &radc_prec);
    *timestamp_ptr = rtimestamp;
    *fcnt_ptr = rfcnt;
    *adcprec_ptr = radc_prec;
    *ds_ptr = ret;
    *tds_ptr = td_size_val;
    *sds_ptr = sd_size_val;


    free(pvalue);
    return 0;
}


int usbfmt_get_one_cone_fullinfo(const std::string &fpath, uint64_t frm_start_pos,
                        py::array_t<int16_t>& raw_np,
                        py::array_t<uint64_t>& timestamp, py::array_t<int>& fcount, py::array_t<int>& adc_bit_prec,
                        int height, int width){


    std::ifstream fin(fpath, std::ios::in | std::ios::binary);
    if(!fin){
        printf("Can not open file %s\n", fpath.c_str());
        return -1;
    }
    int file_size = std::filesystem::file_size(fpath);
    int frame_size = 0;
    fin.seekg(frm_start_pos + 6 * 4);
    fin.read((char *) &frame_size, 4);
    frame_size = frame_size & 0xffffff;
    //printf("cone frame size %d\n", frame_size);
    fin.seekg(frm_start_pos);
    int *pvalue = (int *)calloc(frame_size , sizeof(int));
    fin.read((char *) pvalue, frame_size * sizeof(int));
    fin.close();

    py::buffer_info raw_buf = raw_np.request();
    py::buffer_info timestamp_buf = timestamp.request();
    py::buffer_info fcount_buf = fcount.request();
    py::buffer_info adc_bit_prec_buf = adc_bit_prec.request();

    uint64_t* timestamp_ptr = (uint64_t*) timestamp_buf.ptr;
    int* fcnt_ptr = (int*) fcount_buf.ptr;
    int* adcprec_ptr = (int*) adc_bit_prec_buf.ptr;
    int16_t* raw_ptr = (int16_t*) raw_buf.ptr;

    int ret;
    int fcnt;
    uint64_t ctimestamp;

    uint64_t rtimestamp;
    //int* pval_ptr = pvalue;
    ret = cone_reader_usb_info(pvalue, raw_ptr, height, width, &ctimestamp, &fcnt);
    *timestamp_ptr = ctimestamp;
    *fcnt_ptr = fcnt;
    *adcprec_ptr = 10;
    //rod_decoder_tdsd_size(pvalue, td_ptr, sd_l_ptr, sd_r_ptr, &td_size_val, &sd_size_val, height, width, &rtimestamp, &rfcnt, &radc_prec);
    free(pvalue);
    return 0;
}


bool compareFileName(const std::string& a, const std::string& b) {
    // 提取真实序号
    int aIndex = std::stoi(a.substr(0, a.find('_')));
    int bIndex = std::stoi(b.substr(0, b.find('_')));

    return aIndex < bIndex;
}


void rod_header_repack(std::vector<int> &pkt_buf, int pktptr_this_file){
	// printf("Original: ")	;
	// for(int i = 0;i < 16; i++){
	// 	printf("0x%x ", pkt_buf[i]);
	// }
	// printf("\n");

	pkt_buf[0] = pkt_buf[0] | 0xed800000;

	assert (pkt_buf.size() == pktptr_this_file);
	uint64_t timestamp_high =  (unsigned int)pkt_buf[FRM_HEAD_TimeStampMSB_OFFSET];
	uint64_t timestamp_low = (unsigned int)pkt_buf[FRM_HEAD_TimeStampLSB_OFFSET];
	uint64_t timestamp = ((uint64_t)timestamp_high << 32) + (uint64_t)timestamp_low;
	uint32_t fcnt = pkt_buf[FRM_HEAD_FrmCount_OFFSET];
	// re packt the timestamp
	uint64_t ts_h = static_cast<uint64_t> ((timestamp >> 48) & 0xffffff);
    uint64_t ts_m = static_cast<uint64_t> ((timestamp >> 24) & 0xffffff);
    uint64_t ts_l = static_cast<uint64_t> (timestamp & 0xffffff);
 	uint64_t timestamp_usb = static_cast<uint64_t> ((ts_h) << 48) + (ts_m << 24) + ts_l;
	assert (timestamp_usb == timestamp);
	pkt_buf[1] = ts_h | 0xed000000;
	pkt_buf[2] = ts_m | 0xed000000;
	pkt_buf[3] = ts_l | 0xed000000;
	// re-packet the counter
    uint32_t cnt_h = static_cast<uint32_t> ((fcnt >> 24) & 0xffffff);
    uint32_t cnt_l = static_cast<uint32_t> (fcnt & 0xffffff);
	uint32_t fcnt_usb = static_cast<uint32_t> (cnt_h << 24) + cnt_l;
	assert (fcnt_usb == fcnt);
	pkt_buf[4] = cnt_h | 0xed000000;
	pkt_buf[5] = cnt_l | 0xed000000;
	// add frame size
	pkt_buf[6] = pktptr_this_file | 0xed000000;
	for(int i = 7; i < 16; i++){
		//pkt_buf.push_back(0);
		pkt_buf[i] = pkt_buf[i] | 0xed000000;
	}
	
	// for(int i = 0; i < 16; i++){
	// 	printf("0x%x ", pkt_buf[i]);
	// }
	// printf("\n");
	return;
}

void cone_header_repack(std::vector<int> &pkt_buf, int pktptr_this_file){
	// printf("Original: ")	;
	// for(int i = 0;i < 16; i++){
	// 	printf("0x%x ", pkt_buf[i]);
	// }
	// printf("\n");
	pkt_buf[0] = pkt_buf[0] | 0xfa800000;
	assert (pkt_buf.size() == pktptr_this_file);
	uint64_t timestamp_high =  (unsigned int)pkt_buf[FRM_HEAD_TimeStampMSB_OFFSET];
	uint64_t timestamp_low = (unsigned int)pkt_buf[FRM_HEAD_TimeStampLSB_OFFSET];
	uint64_t timestamp = ((uint64_t)timestamp_high << 32) + (uint64_t)timestamp_low;
	uint32_t fcnt = pkt_buf[FRM_HEAD_FrmCount_OFFSET];
// re packt the timestamp
	uint64_t ts_h = static_cast<uint64_t> ((timestamp >> 48) & 0xffffff);
    uint64_t ts_m = static_cast<uint64_t> ((timestamp >> 24) & 0xffffff);
    uint64_t ts_l = static_cast<uint64_t> (timestamp & 0xffffff);
 	uint64_t timestamp_usb = static_cast<uint64_t> ((ts_h) << 48) + (ts_m << 24) + ts_l;
	assert (timestamp_usb == timestamp);
	pkt_buf[1] = ts_h | 0xfa000000;
	pkt_buf[2] = ts_m | 0xfa000000;
	pkt_buf[3] = ts_l | 0xfa000000;
	// cnt
	uint32_t cnt_h = static_cast<uint32_t> ((fcnt >> 24) & 0xffffff);
    uint32_t cnt_l = static_cast<uint32_t> (fcnt & 0xffffff);
	uint32_t fcnt_usb = static_cast<uint32_t> (cnt_h << 24) + cnt_l;
	assert (fcnt_usb == fcnt);
	pkt_buf[4] = cnt_h | 0xfa000000;
	pkt_buf[5] = cnt_l | 0xfa000000;
	// add frame size
	pkt_buf[6] = pktptr_this_file | 0xfa000000;
	for(int i = 7; i < 16; i++){
		//pkt_buf.push_back(0);
		pkt_buf[i] = pkt_buf[i] | 0xfa000000;
	}
	// for(int i = 0; i < 16; i++){
	// 	printf("0x%x ", pkt_buf[i]);
	// }
	// printf("\n");

	return;
}

int cone_data_repack(int* pvalue, int effect_size, std::vector<int> &raw_buf){
	//header
	for(int i = 0; i < 16; i++){
		raw_buf.push_back(pvalue[i]);
	}

	//data
	int index = 16;
	int data_size = 0;
	for(int i = index; i < effect_size; i += 2){
		int16_t pix = pvalue[i] & 0x3ff;
		int16_t pix2 = pvalue[i+1] & 0x3ff;
		uint32_t two_pix = (pix2 << 10) + pix;
		raw_buf.push_back(two_pix | 0xfe000000);
		data_size ++;

	}
	data_size += 16;

	cone_header_repack(raw_buf, data_size);	

	return data_size;
}
void print_this_rod(std::vector<int> buf){
	for(auto i : buf){
		printf("%x\n", i);
	}
}
void rod_compact_pcie2usb(const std::string &dataset_top, int img_per_file, int size, int one_frm_size, const std::string &save_file_path)
{
    //std::cout<<"The file path is "<<fpath<<"read size"<<size<<"img_per_file "<<img_per_file <<"one_frm_size "<<one_frm_size <<std::endl;
    //     遍历文件夹
	std::string dataset = dataset_top + "/rod";
	std::vector<std::string> fileNames;
	for (const auto& entry : fs::directory_iterator(dataset)) {
       if (fs::is_regular_file(entry) && entry.path().extension() == ".bin") {
            // 获取文件名并添加到 vector
            fileNames.push_back(entry.path().filename().string());
        }
    }
	// Sort all the files
	std::sort(fileNames.begin(), fileNames.end(), compareFileName);
	if(size != img_per_file * one_frm_size){
		std::cout<<"size "<<size<<"img_per_file "<<img_per_file <<"one_frm_size "<<one_frm_size <<std::endl;
		std::cout<<"size may not match because of some GUI bugs..."<<std::endl;
	}
	// interation all files and store to a new file
	std::ofstream newformat_file(save_file_path, std::ios::out | std::ios::binary);
	int *pvalue = (int *)calloc(size , sizeof(int));
	for (const auto& fileName : fileNames) {
			//std::cout << fileName << std::endl;
		std::string datafile = dataset + "/" + fileName;
		std::ifstream fin(datafile, std::ios::in | std::ios::binary);
		fin.read((char *) pvalue, size * sizeof(int));
		fin.close();

		int f_offset = one_frm_size;
		int frm_size_this = 0;
		
		for(int i = 0; i < img_per_file; i++){
//         //  std::cout<<"processing "<<i<<" th img"<<std::endl;
			int* pval_ptr = pvalue + f_offset * i;
			int index = 0;
			//int pktptr_this_frm = 0;
			std::vector<int> pkt_buf;
			while(true){
				//  unsigned int pkt = (unsigned int) pval_ptr[index];
				if((unsigned int) pval_ptr[index] !=  0xffffffff){//end of one frame packet 
					//pkt_buf_ptr[pktptr_this_file] = pvalue[index];
					pkt_buf.push_back(pval_ptr[index]);
					frm_size_this++;
					index++;
					
				}else{
					pkt_buf.push_back(0xffffffff);
					frm_size_this++;
					rod_header_repack(pkt_buf, frm_size_this);
					//print_this_rod(pkt_buf);
					newformat_file.write((char *) pkt_buf.data(), frm_size_this * sizeof(int));
					frm_size_this = 0;
					//pkt_buf_ptr[pktptr_this_file - index] |=  (index << 4) | 0x80000000; // packet num in this frame
					//printf("packt num in %d frame: %d, pkt_header 0x%x\n", i, index, pkt_buf_ptr[pktptr_this_file - index]);
					//total_pkt_num += index;
					break;
				}
			}
        // int8_t* td_val = td_ptr + diffoffset * i;
   	 }
	}
	newformat_file.close();
	free(pvalue);

    return;
}

void cone_compact_pcie2usb(const std::string &dataset_top, int effect_size, const std::string &save_file_path){
	std::string dataset = dataset_top + "/cone";
	std::vector<std::string> fileNames;
	for (const auto& entry : fs::directory_iterator(dataset)) {
       if (fs::is_regular_file(entry) && entry.path().extension() == ".bin") {
            // 获取文件名并添加到 vector
            fileNames.push_back(entry.path().filename().string());
        }
    }
	// Sort all the files
	std::sort(fileNames.begin(), fileNames.end(), compareFileName);

	// interation all files and store to a new file
	std::ofstream newformat_file(save_file_path, std::ios::out | std::ios::binary);

	int *pvalue = (int *)calloc(effect_size , sizeof(int));
	for (const auto& fileName : fileNames) {
			//std::cout << fileName << std::endl;
		std::string datafile = dataset + "/" + fileName;
		std::ifstream fin(datafile, std::ios::in | std::ios::binary);
		fin.read((char *) pvalue, effect_size * sizeof(int));
		fin.close();
		std::vector<int> raw_buf;
		int frm_size_this = cone_data_repack(pvalue, effect_size, raw_buf);
		newformat_file.write((char *) raw_buf.data(), frm_size_this * sizeof(int));


	}
	newformat_file.close();
	free(pvalue);
	return;
}


// int usbfmt_get_multi_rod_fullinfo(const std::string &fpath, int frm_start_pos, int read_num, 
//                         py::array_t<int8_t>& temp_diff_np, py::array_t<int8_t>& spat_diff_left_np, py::array_t<int8_t>& spat_diff_right_np, 
//                         py::array_t<int>& diff_size, py::array_t<int>& td_size, py::array_t<int>& sd_size,
//                         py::array_t<uint64_t>& timestamp, py::array_t<int>& fcnt, py::array_t<int>& adc_bit_prec,
//                         int height, int width){


    
//     std::ifstream fin(fpath, std::ios::in | std::ios::binary);
//     if(!fin){
//         printf("Can not open file %s\n", fpath.c_str());
//         return -1;
//     }
//     //int file_size = getFileSize(fpath.c_str());
//     int fileSize = static_cast<int>file.tellg();
//     int max_size = 40448;
//     int *pvalue = (int *)calloc(max_size , sizeof(int));
//     fin.read((char *) pvalue, frame_size * sizeof(int));
//     py::buffer_info td_buf = temp_diff_np.request();
//     py::buffer_info sd_l_buf = spat_diff_left_np.request();
//     py::buffer_info sd_r_buf = spat_diff_right_np.request();
//     py::buffer_info ds_buf = diff_size.request();
//     py::buffer_info tds_buf = td_size.request();
//     py::buffer_info sds_buf = sd_size.request();
//     py::buffer_info timestamp_buf = timestamp.request();
//     py::buffer_info fcount_buf = fcnt.request();
//     py::buffer_info adc_bit_prec_buf = adc_bit_prec.request();

//   uint64_t* timestamp_ptr = (uint64_t*) timestamp_buf.ptr;
//     int* fcnt_ptr = (int*) fcount_buf.ptr;
//     int* adcprec_ptr = (int*) adc_bit_prec_buf.ptr;
//     //int* pvalue = (int*) malloc(pvalue_np.size());
//     //Py_ssize_t size = PyBytes_GET_SIZE(pvalue.ptr());
//     //int* pval_ptr = (int*) pvalue.ptr();
//    // int* pval_ptr = (int *) pvalue;
//     int diffoffset = width * height;
    
//     int8_t* td_ptr = (int8_t*) td_buf.ptr;
//     int8_t* sd_l_ptr = (int8_t*) sd_l_buf.ptr;
//     int8_t* sd_r_ptr = (int8_t*) sd_r_buf.ptr;
//     int* ds_ptr = (int*) ds_buf.ptr;
//      int* tds_ptr = (int*) tds_buf.ptr;
//      int* sds_ptr = (int*) sds_buf.ptr;
//     int reader_frm_start_pos = frm_start_pos;
//     for (int i = 0; i < read_num; i++){
//         if(reader_frm_start_pos >= file_size){
//             break;
//         }
//         fin.seekg(reader_frm_start_pos + 6 * 4);
//         fin.read((char *) &frame_size, 4);
//         fin.seekg(reader_frm_start_pos);
//         int *pvalue = (int *)calloc(frame_size , sizeof(int));
//         fin.read((char *) pvalue, frame_size * sizeof(int));
//         reader_frm_start_pos = frame_size * 4 + reader_frm_start_pos;

    
//         int td_size_val, sd_size_val;
//         int radc_prec, rfcnt;
//         uint64_t rtimestamp;
//         //int* pval_ptr = pvalue;
//         int ret = rod_decoder_tdsd_size(pvalue, td_ptr, sd_l_ptr, sd_r_ptr, &td_size_val, &sd_size_val, height, width, &rtimestamp, &rfcnt, &radc_prec);
//      }
//     fin.close();
//     free(pvalue);
        

//     return 0;
// }

PYBIND11_MODULE(rod_decoder_py, m){
    m.doc() = "pybind11 rod decoder";
    m.def("rod_decoder_py", &rod_decoder_py, "rod_decoder_py" );
    //m.def("rod_decoder_py_bulk", &rod_decoder_py_bulk, "rod_decoder_py_bulk" );
    m.def("rod_decoder_py_byfile", &rod_decoder_py_byfile, "rod_decoder_py_byfile" );
    m.def("cone_reader_py_byfile", &cone_reader_py_byfile, "cone_reader_py_byfile" );
    m.def("rod_decoder_py_byfile_bw", &rod_decoder_py_byfile_bw, "rod_decoder_py_byfile_bw" );
    m.def("rod_decoder_py_byfile_td_sd_bw", &rod_decoder_py_byfile_td_sd_bw, "rod_decoder_py_byfile_td_sd_bw" );
   // m.def("rod_compact_pkt", &rod_compact_pkt, "rod_compact_pkt");
    m.def("rod_decoder_py_fullInfo", &rod_decoder_py_fullInfo, "rod_decoder_py_fullInfo");
    m.def("cone_reader_py_fullInfo", &cone_reader_py_fullInfo, "cone_reader_py_fullInfo");
    m.def("rod_decoder_py_onlyinfo", &rod_decoder_py_onlyinfo, "rod_decoder_py_onlyinfo");
    m.def("construct_frm_list", &usbfmt_contruct_frm_list, "usbfmt_contruct_frm_list");
    m.def("get_one_rod_fullinfo", &usbfmt_get_one_rod_fullinfo, "usbfmt_get_one_rod_fullinfo");
     m.def("get_one_cone_fullinfo", &usbfmt_get_one_cone_fullinfo, "USB format get one cone frame with full info");
    m.def("rod_pcie2usb_conv", &rod_compact_pcie2usb, "Convert Rod PCIE data to USB data format");
    m.def("cone_pcie2usb_conv", &cone_compact_pcie2usb, "Convert Cone PCIE data to USB data format");

  
}
