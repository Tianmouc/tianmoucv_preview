#include <filesystem>
#include <iostream>
#include <fstream>
#include <getopt.h>

/************   Variable definations for single camera! **************/
#define ROD_8B_ONE_FRM 0x9e00       // 158KB * 1024 / 4;//0x9e00
    //last frame address in DDR
#define ROD_4B_ONE_FRM 0x4D00 //
//define ROD_2B_ONE_FRM 0x1C40
//#define ROD_2B_LAST_FRM_ADDR 0x27fffcb00
#define ROD_2B_ONE_FRM 0x1D00

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
    std::string dataset = dataset_top + "/rod";
    int size = one_frm_size * img_per_file;
    std::ifstream fin(dataset, std::ios::in | std::ios::binary);
    int *pvalue = (int *)calloc(size , sizeof(int));
    fin.read((char *) pvalue, size * sizeof(int));
    fin.close();



}