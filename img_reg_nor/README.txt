# img_reg_nor README.txt note
# 12-16-2024 in Korea Roy Seo

관련 scripts, files & folders
3-3-run_fs_modified_plus_noramalized2mni152_1mm_fsl_flirt_linear_transform.sh 
4_Tina_adni_reg_nor_RSedits_12_11_24_subj111_4110.sh  
/ibic/scratch/royseo_workingdir/normalized2mni152_1mm 


1. 처음에 Tina와 10개 subject로 실험
  - file: sub_list_ADNI1234_1-10.log
  - commend: qsub -q all.q 4_Tina_adni_reg_nor_RSedits_12_07_24_subj1-10.sh 

2. 잘 되는 것 확인하고 4000개 subjects 를 돌리기로 함
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
