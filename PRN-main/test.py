import sys
from subprocess import call
import os
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


if __name__ == '__main__':

    img=os.listdir('testimg')
    mask=os.listdir('mask')
    img=img[0]
    mask=mask[0]

    # Stage1: brightness map generation
    os.chdir("brightness")
    stage_1_command="python run.py --data_root ./testimg/1.png --mask_root ./mask/1.png --model_path checkpoint/brightness.pth --test --mask_mode 0"
    run_cmd(stage_1_command)
    print("Finish the Stage 1 - Brightness map has been completed, please check the folder result1")

    ### Stage2: Grayscale map generation
    os.chdir("../sketch")
    # stage_2_command = "python run.py --data_root ./testimg --mask_root ./mask --model_path checkpoint/g_940000.pth --test --mask_mode 0"
    stage_2_command = "python run.py --data_root ./testimg/1.png --mask_root ./mask/1.png --model_path checkpoint/g_940000.pth --test --mask_mode 0"
    run_cmd(stage_2_command)
    print("Finish the Stage 2 - Sketch map has been completed, please check the folder result2")
    #
    ### Stage3: color restoration
    os.chdir("../PIAFusion_pytorch-masterGPU3")
    shutil.copy("../mask/"+mask,"../colorrestoredata/originalmask/")
    shutil.copy("../result1/results/img_1.png", "../colorrestoredata/bright/1.png")
    shutil.copy("../result2/results/img_1.png", "../colorrestoredata/cartoonresult/1.png")
    shutil.copy("../testimg/"+img, "../colorrestoredata/originalimage/1.png")

    stage_3_command = "python test_fusion_model.py --dataset_path ../colorrestoredata --save_path ../finalresult --fusion_pretrained pretrained/fusion_model_epoch_9.pth"
    run_cmd(stage_3_command)
    print("Finish the Stage 3 - All steps have been completed, please check the folder finalresult")





    ### Stage3: color restoration
    # os.chdir("../rgb_restore")
    #
    # stage_3_command = "python main.py  --img_path ../testimg/"+img+" --gray_path ../result1/results/img_1.png --color_path ../result2/results/img_1.png --mask_path ../mask/"+mask
    # run_cmd(stage_3_command)
    # print("Finish the Stage 3 - All steps have been completed, please check the folder resultfinal")





