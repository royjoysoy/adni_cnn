# img_reg_nor README.txt note
# 12-16-2024 in Korea Roy Seo

관련 scripts, files & folders
3-3-run_fs_modified_plus_noramalized2mni152_1mm_fsl_flirt_linear_transform.sh 
4_Tina_adni_reg_nor_RSedits_12_11_24_subj111_4110.sh  
/ibic/scratch/royseo_workingdir/normalized2mni152_1mm 


1. 처음에 Tina와 10개 subject로 실험
  - file: sub_list_ADNI1234_1-10.log
  - commend: qsub -q all.q 4_Tina_adni_reg_nor_RSedits_12_07_24_subj1-10.sh 

2. 잘 되는 것 확인하고 4000개 subjects 를 돌리기로 함 : 100개만 돌아감
- file: subj_list_ADNI1234_11-4010.log
- commend: qsub -q all.q 4_Tina_adni_reg_nor_RSedits_12_XX_24_subj11-4010.sh 
         : 4_Tina4_Tina_adni_reg_nor_RSedits_12_XX_24_XXXXX
         : 후에 오버라잇 되었기때문에 정확한 날짜와 (그래서 XX라고 씀)) 아마 subj11-4010.sh는 이라고 예상
         : github뒤져보면 정확히 알 수 있을듯 
 
7번째 line #s -t 1-4000이어야 되는데 core 100개를 recruit한다고 생각하여 1-100이라고 썼고 
그래서 100개의 subjects만 돌아갔다고 Tina에게 연락옴 Dec 11 8: 56 AM (In Korea time)

3. 12/11 수요일 12:31pm 4000개 subjects .log 파일 만들어서 다시 돌리기 시작 
- normalized2mni152_1mm_11_110 이름 다시 만들고 새롭게 normalized2mni152_1mm만들고 돌림 
- file: subj_list_ADNI1234_28001_111-4110.log
- commend: qsub -q all.q 4_Tina_adni_reg_nor_RSedits_12_11_24_subj111_4110.sh

4. 12/16/24
- found out the mismatch between royseo@adrc.ibic.washington.edu:/ibic/scratch/royseo_workingdir/scripts/
- conti. and my local computer ~/Desktop/adni_cnn/img_reg_nor folder
- conti.내가 100개만 돌아간것 알고, 전에 4000개를 돌리려고 해놓았던 스크립트나 subj_list_ADNI1234_28001_111-4110.log파일을 
- conti. update했었는데 그것을  adrc.ibic.washington.edu:/ibic/scratch/royseo_workingdir/scripts에서 옮겼음
- local computer 에서 사용했던 명령어: 
scp royseo@adrc.ibic.washington.edu:/ibic/scratch/royseo_workingdir/scripts/{4_Tina_adni_reg_nor_RSedits_12_07_24_subj1-10.sh,subj_list_ADNI1234_28001_11-4010.log} ~/Desktop/adni_cnn/img_reg_nor/


5. 12/23/24
- subj_list_ADNI11234_28001_4111-6000.log에서 11subjects 안 돌아간것 qsub -q all.q 4_Tina_adni_reg_nor_RSedits_12_22_24_subj4111_6000_troubleshoot.sh로 돌리기 시작 12/23/24 12:43pm (Korea Time)
- 자세한 내용 log 는 "subj_list_ADNI1234_28001_4111-6000_troubleshoot.txt" 이 파일에서 확인
- "3-3-run_fs_modified_plus_noramalized2mni152_1mm_fsl_flirt_linear_transform.sh" 파일 이름에 noramalized 오타 발견 그래서 고침 -> 3-3-run_fs_modified_plus_normalized2mni152_1mm_fsl_flirt_linear_transform.sh
- normalize2mni152_1mm (output 폴더) -> normazlied2mni152_1mm_4111-6000wo11으로 바꿈
- normazlied2mni152_1mm_4111-6000wo11 폴더는 사실 1890개 즉 trouble났던 다음의 11 subjects를 포함하고 있음
016_S_0702_2013-05-09_S18903_I377040.nii
021_S_0337_2012-04-24_S14825_I327057.nii
021_S_0337_2012-04-24_S14825_I327059.nii
021_S_0337_2012-04-24_S14825_I327062.nii
021_S_0337_2012-04-24_S14825_I327074.nii
021_S_0337_2013-05-07_S18878_I389140.nii
021_S_0337_2013-05-07_S18879_I389138.nii
021_S_0337_2014-04-24_S21696_I432139.nii
021_S_0337_2014-04-24_S21696_I432140.nii
021_S_0337_2015-05-07_S25882_I510022.nii
021_S_0337_2015-05-07_S25882_I510051.nii

-??? 이해 안가는것 ???
- qsub -q all.q 4_Tina_adni_reg_nor_RSedits_12_22_24_subj4111_6000_troubleshoot.sh을 돌렸는데 
이것은 qsub -q all.q 4_Tina_adni_reg_nor_RSedits_12_22_24_subj4111_6000_troubleshoot.sh 스크립트를 돌리고 이 스크립트는 output folder 를
export SUBJECTS_DIR="//ibic/scratch/royseo_workingdir/normalized2mni152_1mm_subj4111_6000_troubleshoot 이렇게 지정함
그래서 normalized2mni152_1mm_subj4111_6000_troubleshoot를 만들어놨는데 거기에 output이 저장되는 것이 아니고 normalized2mni152_1mm 폴더를 만들어 거기에 저장되고 있음




