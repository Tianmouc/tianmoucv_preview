#include <filesystem>
#include <iostream>
#include <fstream>
#include <getopt.h>
#include <cassert>

///#define __cplusplus 201705L
#define FRM_HEAD_OFFSET 16
#define FRM_HEAD_CNT_TS_OFFSET 4
const uint frm_head_cnt_ts_offset = FRM_HEAD_CNT_TS_OFFSET;
#define FRM_HEAD_ADC_PREC_OFFSET 0
#define FRM_HEAD_FrmCount_OFFSET 3
#define FRM_HEAD_TimeStampMSB_OFFSET 1
#define FRM_HEAD_TimeStampLSB_OFFSET 2
#define FRM_HEAD_READOUT_FLAG_OFFSET 0
#define FRM_HEAD_RAND_PATTERN_OFFSET 12
#define CONE_FRM_HEAD_TimeStampMSB_OFFSET 1
#define CONE_FRM_HEAD_TimeStampLSB_OFFSET 2
#define CONE_FRM_HEAD_FrmCount_OFFSET 3

#include <vector>
#include <algorithm>
/************   Variable definations for single camera! **************/
#define ROD_8B_ONE_FRM 0x9e00       // 158KB * 1024 / 4;//0x9e00
    //last frame address in DDR
#define ROD_4B_ONE_FRM 0x4D00 //
//define ROD_2B_ONE_FRM 0x1C40
//#define ROD_2B_LAST_FRM_ADDR 0x27fffcb00
#define ROD_2B_ONE_FRM 0x1D00

namespace fs = std::filesystem;

static struct option const long_opts[] = {
	{"adc_bit_prec", required_argument, NULL, 'p'},
	{"img_per_file", required_argument, NULL, 'i'},
	{"dataset_top", required_argument, NULL, 'd'},
	{"help", no_argument, NULL, 'h'},
	{"verbose", no_argument, NULL, 'v'},
	{0, 0, 0, 0}
};

void helpword(){
    printf("this is help\n");
    return;
}



void rod_compact_pcie2usb(const std::string &dataset_top, int img_per_file, int size, int one_frm_size, const std::string &save_file_path);
void cone_compact_pcie2usb(const std::string &dataset_top, int effect_size, const std::string &save_file_path);

int main(int argc, char *argv[]){
    std::string dataset_top;
    int adc_bit_prec = 0;
    int img_per_file = 0;
    int cmd_opt;
    bool verbose;
    while ((cmd_opt =
		getopt_long(argc, argv, "vh:p:i:d:", long_opts,
			    NULL)) != -1) {
		switch (cmd_opt) {
		case 0:
			/* long option */
			break;
		case 'd':
			/* dataset top  */
			//fprintf(stdout, "'%s'\n", optarg);
			dataset_top = optarg;
			break;
		case 'p':
			/* ADC bit Precision */
			adc_bit_prec = atoi(optarg);
			break;
		case 'i':
			/* image per file */
			img_per_file = atoi(optarg);
			break;

		case 'v':
			verbose = true;
			break;
		case 'h':
		default:
			helpword();
			exit(0);
			break;
		}
	}
	//only for test
	adc_bit_prec = 8;
	img_per_file = 2;
	dataset_top = "/data/taoyi/dataset/Lyncam/2023_02_16_zy/traditional/simens_c1800r400";
    int one_frm_size;

    switch (adc_bit_prec)
    {
    case 2:
        one_frm_size = ROD_2B_ONE_FRM;
        break;
    case 4:
        one_frm_size = ROD_4B_ONE_FRM;
        break;
    case 8:
        one_frm_size = ROD_8B_ONE_FRM;
        break;        
    default:
        printf("wrong ADC prec %u bit, program exit \n",adc_bit_prec);
        exit(-1);
        break;
    }
	int size = one_frm_size * img_per_file;	
	std::string save_file_path =dataset_top +  "/rod_compact.tmdat";
	rod_compact_pcie2usb(dataset_top,  img_per_file, size,  one_frm_size, save_file_path);
	std::string cone_save_file_path =dataset_top +  "/cone_compact.tmdat";
	int cone_eff_size = 102400 + 16;
	cone_compact_pcie2usb(dataset_top, cone_eff_size, cone_save_file_path);
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
	uint64_t timestamp_high =  (uint)pkt_buf[FRM_HEAD_TimeStampMSB_OFFSET];
	uint64_t timestamp_low = (uint)pkt_buf[FRM_HEAD_TimeStampLSB_OFFSET];
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
void rod_data_repack(){
	return;
}
void cone_header_repack(std::vector<int> &pkt_buf, int pktptr_this_file){
	printf("Original: ")	;
	for(int i = 0;i < 16; i++){
		printf("0x%x ", pkt_buf[i]);
	}
	printf("\n");
	pkt_buf[0] = pkt_buf[0] | 0xfa800000;
	assert (pkt_buf.size() == pktptr_this_file);
	uint64_t timestamp_high =  (uint)pkt_buf[FRM_HEAD_TimeStampMSB_OFFSET];
	uint64_t timestamp_low = (uint)pkt_buf[FRM_HEAD_TimeStampLSB_OFFSET];
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
	for(int i = 0; i < 16; i++){
		printf("0x%x ", pkt_buf[i]);
	}
	printf("\n");

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
				//  uint pkt = (uint) pval_ptr[index];
				if((uint) pval_ptr[index] !=  0xffffffff){//end of one frame packet 
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



// 	printf("read from %s\n", datafile.c_str());

//     std::ifstream fin(datafile, std::ios::in | std::ios::binary);
//     int *pvalue = (int *)calloc(size , sizeof(int));
//     fin.read((char *) pvalue, size * sizeof(int));
// 	for (int i = 0; i < size ; i++){
// 		printf("%x\n", (uint) pvalue[i]);
// 	}
//     fin.close();


	
//     int size = one_frm_size * img_per_file;

// 	std::vector<std::string> flist;
//      for (const std::string& str : flist) {
//         std::cout << str << std::endl;
// 		std::string datafile = dataset + "/102_592887.bin";//"/100_592808.bin";
//     }


//     std::ifstream fin(fpath, std::ios::in | std::ios::binary);
//     int *pvalue = (int *)calloc(size , sizeof(int));
//     fin.read((char *) pvalue, size * sizeof(int));
//     fin.close();
//      //int *rod_pkt = (int *)calloc(size , sizeof(int));//compact_pkt
//     //py::buffer_info pvalue_buf = pvalue_np.request();// 获取arr1, arr2的信息
//     py::buffer_info pkt_buf = rod_pkt_np.request();
   
//    // int diffoffset = width * height;
//     int f_offset = one_frm_size;
//     int32_t* pkt_buf_ptr = (int32_t*) pkt_buf.ptr;

//     //int ret;
    
//     int total_pkt_num = 0;
//     int pktptr_this_file = 0;
//     for(int i = 0; i < img_per_file; i++){
//         //  std::cout<<"processing "<<i<<" th img"<<std::endl;
//             int* pval_ptr = pvalue + f_offset * i;
//             int index = 0;
//             //int pktptr_this_frm = 0;
//             while(true){
//               //  uint pkt = (uint) pval_ptr[index];
//                 if((uint) pval_ptr[index] !=  0xffffffff){//end of one frame packet 
//                     pkt_buf_ptr[pktptr_this_file] = pvalue[index];
//                     pktptr_this_file++;
//                     index++;
//                 }else{
//                     pkt_buf_ptr[pktptr_this_file - index] |=  (index << 4) | 0x80000000; // packet num in this frame
//                     printf("packt num in %d frame: %d, pkt_header 0x%x\n", i, index, pkt_buf_ptr[pktptr_this_file - index]);
//                     total_pkt_num += index;
//                     break;
//                 }
//             }
//         // int8_t* td_val = td_ptr + diffoffset * i;
//     }


//    free(pvalue);
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