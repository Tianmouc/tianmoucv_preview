
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cstdio>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

//just for test

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
/************   Variable definations for single camera! **************/
#define ROD_8B_ONE_FRM 0x9e00       // 158KB * 1024 / 4;//0x9e00


    //last frame address in DDR
#define ROD_4B_ONE_FRM 0x4D00 //

//define ROD_2B_ONE_FRM 0x1C40
//#define ROD_2B_LAST_FRM_ADDR 0x27fffcb00
#define ROD_2B_ONE_FRM 0x1D00



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



PYBIND11_MODULE(rod_decoder_py, m){
    m.doc() = "pybind11 rod decoder";
    m.def("rod_decoder_py", &rod_decoder_py, "rod_decoder_py" );
    //m.def("rod_decoder_py_bulk", &rod_decoder_py_bulk, "rod_decoder_py_bulk" );
    m.def("rod_decoder_py_byfile", &rod_decoder_py_byfile, "rod_decoder_py_byfile" );
    m.def("cone_reader_py_byfile", &cone_reader_py_byfile, "cone_reader_py_byfile" );
    m.def("rod_decoder_py_byfile_bw", &rod_decoder_py_byfile_bw, "rod_decoder_py_byfile_bw" );
    m.def("rod_decoder_py_byfile_td_sd_bw", &rod_decoder_py_byfile_td_sd_bw, "rod_decoder_py_byfile_td_sd_bw" );
    m.def("rod_compact_pkt", &rod_compact_pkt, "rod_compact_pkt");
  
}
