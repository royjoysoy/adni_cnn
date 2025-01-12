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

6. 12/30/24
- 12/29/24일날 만들어 두었던 2 files: 
  1. 4_Tina_adni_reg_nor_RSedits_12_30_24_subj12001_18000.sh
  2. subj_list_ADNI1234_28001_12001-18000.log 
  을 돌렸음
- normalize2mni152_1mm 폴더를 normalize2mni152_1mm_6001-12000wo12 로 이름을 바꿈
- normalize2mni152_1mmwo12는 12개의 subj folders가 folders는 있으나 brain.mgz가 만들어지지 않았기 때문에
  그 다음 단계인 mni152_1mm로 normalize되는 것도 되지 않았음
- 그 12개의 folders:

021_S_0984_2014-12-09_S24277_I467254 not complete
023_S_0625_2007-07-16_S35036_I73330 not complete
023_S_0625_2008-01-11_S44497_I90886 not complete
023_S_0625_2008-01-11_S44498_I90882 not complete
023_S_0625_2008-09-29_S56786_I124075 not complete
023_S_0625_2008-09-29_S56787_I124080 not complete
027_S_0644_2012-07-02_S15633_I334521 not complete
027_S_4964_2012-10-06_S17005_I358248 not complete
031_S_0294_2007-09-25_S40106_I79604 not complete
031_S_0618_2006-06-06_S15271_I67110 not complete
031_S_0618_2007-07-02_S36607_I72188 not complete
033_S_0513_2007-05-31_S33069_I67706 not complete



- 한국시간 12/30/24 월요일 낮 12:05분에 다음을 submitted!
  qsub -q all.q 4_Tina_adni_reg_nor_RSedits_12_30_24_subj12001_18000.sh
- qstat -u royseo 해 본 결과 qsub이 잘 submitted 된 것을 확인
- 182개의 폴더가 바로 생겼음 (Does it mean that qsub commend recruit 182 cores or 
  one core recruits can make a few folders simultaneously?) 

- 그 전에 돌린 60001-12000 subjects 중 다음의 12개 subjects는 brain.mgz가 만들어지지 않았음

7. 1/1/25
- 아직 4_Tina_adni_reg_nor_RSedits_12_30_24_subj12001_18000.sh 파일이 다 돌지는 않았음
- Next two files 만듦
    1. subj_list_ADNI1234_28001_18001-28001.log 
    2. 4_Tina_adni_reg_nor_RSedits_01_09_25_subj18001_28001.sh

8. 1/2/25
- 12/22/24:  seoroy15@gmail.com 김대진 박사님이 fnirt을 HCP parameters를 참고하여 해보는게 어떻냐고 하셔서 
- https://github.com/Washington-University/HCPpipelines/blob/master/PreFreeSurfer/PreFreeSurferPipeline.sh 사용하여 fnirt을 시도함
- claude project names: adni-cnn-normalization
- fnirt을 위해 만든 script: 3-4-fnirt-nonlinear-HCP-param.sh

9. 1/3/25
- 3-4-fnirt-nonlinear-HCP-param-batch-processing.sh: 
  1. chaged output directory (fnirt_output -> normalized2mni152_1mm/subject각자/mni152_1mm 로 바꿈)
  2. error logging files 이름에 qsub task ID와 subjectID 를 같이 넣음 
- 4-1-fnirt-nonlinear-HCP-param-batch-processing-submit-fnirt-jobs.sh
  3-4-fnirt-nonlinear-HCP-param-batch-processing.sh의 2번째꺼를 고치면서 fnirt_output 폴더 만드는 부분 없앰

10. 1/4/25
- 1-10 subjects돌려보고 잘되면 11-6000subject qsub으로 돌릴 5개의 파일들 만들음 
- normalized2mni152_1mm_11-6000 폴더 만들고, 
    normalized2mni152_1mm_11-110,
    normalized2mni152_1mm_111-4110,
    normalized2mni152_1mm_4111-6000wo11 이 세 폴더에서 subjects 폴더들 다 모았음
참고로 normalized2mni152_1mm_4111-6000wo11 폴더에는 4111-6000까지 다 있다. 11개가 error 나서 brain.mgz파일부터 생성이 안되었지만, subjects 폴더는 다 있다. 
- 1-10: 
  1.3-4-fnirt-nonlinear-HCP-param-batch-processing-1-10.sh
  2. 4-1-fnirt-nonlinear-HCP-param-batch-processing-submit-fnirt-jobs-1-10.sh
   log file은 이미 있던 것  subj_list_ADNI1234_28001_1-10.log 사용
- 11-6000:
  3. 3-4-fnirt-nonlinear-HCP-param-batch-processing-11-6000.sh  
  4. 4-1-fnirt-nonlinear-HCP-param-batch-processing-submit-fnirt-jobs-11-6000.sh
  5. subj_list_ADNI1234_28001_11-6000_fnirt.log

11. 1/6/25
- fnirt 1-10, 11-6000까지 돌렸음 새벽 2:30분쯤 시작해서 그 다음날 저녁 6:00좀 넘어서 끝났음)
- fnirt중에 error 난 것들 (1-10까지는 에러 없고, 11-6000까지는 에러 있음)
016_5_0702 2013-05-09_518903_1377040 not complete
021_5_0337_2012-04-24_S14825_I327057 not complete
021_5_0337_2012-04-24_S14825_I327059 not complete
021_5_0337_2012-04-24_S14825_I327062 not complete
021_5_0337_2012-04-24_S14825_I327074 not complete
021_5_0337_2013-05-07_S18878_1389140 not complete
021_5_0337_2013-05-07_S18879_I389138 not complete
021_5_0337_2014-04-24_S21696_I432139 not complete
021_5_0337_2014-04-24_S21696_I432140 not complete
021_5_0337_2015-05-07_S25882_I510022 not complete
021_5_0337_2015-05-07_S25882_I510051 not complete

- fnirt 6000-12000까지 돌리기 시작함 6:15pm 아래 두개의 파일로
  1. 3-4-fnirt-nonlinear-HCP-param-batch-processing-6001-12000.sh
  2. 4-1-fnirt-nonlinear-HCP-param-batch-processing-submit-fnirt-jobs-6001-12000.sh

- fnirt 12001-18000 돌릴 파일 만들었음 
  1. 3-4-fnirt-nonlinear-HCP-param-batch-processing-12001-18000.sh
  2. 4-1-fnirt-nonlinear-HCP-param-batch-processing-submit-fnirt-jobs-12001-18000.sh

12. 1/7/25 
-  1. 3-4-fnirt-nonlinear-HCP-param-batch-processing-6001-12000.sh;   4-1-fnirt-nonlinear-HCP-param-batch-processing-submit-fnirt-jobs-6001-12000.sh
이 두 파일 다 돌아 갔음 10:34 am쯤
- 1. 3-4-fnirt-nonlinear-HCP-param-batch-processing-12001-18000.sh;   4-1-fnirt-nonlinear-HCP-param-batch-processing-submit-fnirt-jobs-12001-18000.sh
이 두 파일 돌리기 시작 1:00pm 쯤


13. 1/8/25
  -   qsub -q all.q 4_Tina_adni_reg_nor_RSedits_01_08_25_subj18001_28001.sh  
  10:51pm돌리기 시작
      
14. 1/9/25
*** Important ****
Error 정리
 - 1. 28002개 다운 받음
   - 로그인 후 https://ida.loni.usc.edu/pages/access/search.jsp?project=ADNI&tab=collection&page=SEARCH&subPage=NEW_ADV_QUERY 
          - 위의 주소는 Search & Download -> Advanced Image Search -> Data Collections -> My Collections + -> ADNI_1_2_3_4_11/14/24(28002) 하면 나오는 곳
    -CSV: raw: /adni_cnn/behavioral/raw_28002_ADNI_1_2_3_4_11_14_24_1_08_2025.csv  
          - 위의 파일은 , Subject, Group (initial visit dx, 그래서 dx가 visit이 바뀌어도 변하지 않는다, 
                       Sex, Age, Visit (sc: screening, m: month, bl: baseline, v:찾아봐야함), Modality)
                       Description, Type, Acq Date, Format, Downloaded 가 있는데 다운받으면 Image Data ID 칼럼도 생김
    
 - 2. /ibic/scratch/ADNI (1189의 subject folders, Tina가 Download해줌) 
        /ibic/scratch/royseo_workingdir/raw (28002개의 images: 1189개의 subjects 폴더에서 이미지만 꺼내어 정리)

 - 3. flirt한 결과
      /ibic/scratch/royseo_workingdir/normalized2mni152_1mm_1-10:          (n = 10    | image (n = 10)) 
      /ibic/scratch/royseo_workingdir/normalized2mni152_1mm_11-110:        (n = 101   | image (n = 100) + errorlog (n = 1))
      /ibic/scratch/royseo_workingdir/normalized2mni_1mm_111-4110:         (n = 4000  | image (n = 4000))
      /ibic/scratch/royseo_workingdir/normalized2mni152_1mm_4111_6000wo11: (n = 1890  | image (n = 1890)): file 이름에 'wo11'이 헷갈릴 수 있는데 error가 난 11개의 폴더도 포함되어있다. 'wo11'은 에러난 11개 폴더안에 최종 파일인 "....brain_mni152_1mm.nii.gz" 에러가 나서 안생겼다는 뜻
            1  016_S_0702_2013-05-09_S18903_I377040 not complete
            2  021_S_0337_2012-04-24_S14825_I327057 not complete
            3  021_S_0337_2012-04-24_S14825_I327059 not complete
            4  021_S_0337_2012-04-24_S14825_I327062 not complete
            5  021_S_0337_2012-04-24_S14825_I327074 not complete
            6  021_S_0337_2013-05-07_S18878_I389140 not complete
            7  021_S_0337_2013-05-07_S18879_I389138 not complete
            8  021_S_0337_2014-04-24_S21696_I432139 not complete
            9  021_S_0337_2014-04-24_S21696_I432140 not complete
            10 021_S_0337_2015-05-07_S25882_I510022 not complete
            11 021_S_0337_2015-05-07_S25882_I510051 not complete

      /ibic/scratch/royseo_workingdir/normalized2mni152_1mm_6001_12000wo12 (n = 6000  | image (n = 6000)): file 이름에 'wo12'이 헷갈릴 수 있는데 error가 난 11개의 폴더도 포함되어있다. 'wo12'은 에러난 12개 폴더안에 최종 파일인 "....brain_mni152_1mm.nii.gz" 에러가 나서 안생겼다는 뜻
            1  021_S_0984_2014-12-09_524277_I467254 not complete
            2  023_S_0625_2007-07-16_535036_173330 not complete
            3  023_S_0625_2008-01-11_544497_190886 not complete
            4  023_S_0625_2008-01-11_544498_190882 not complete
            5  023_S_0625_2008-09-29_556786_I124075 not complete
            6  023_S_0625_2008-09-29_556787_I124080 not complete
            7  027_S_0644_2012-07-02_S15633_I334521 not complete
            8  027_S_4964_2012-10-06_S17005_I358248 not complete
            9  031_S_0294_2007-09-25_540106_I79604 not complete
            10 031_S_0618_2006-06-06_S15271_I67110 not complete
            11 031_S_0618_2007-07-02_536607_I72188 not complete
            12 033_S_0513_2007-05-31_S33069_I67706 not complete

      /ibic/scratch/royseo_workingdir/normalized2mni152_1mm_12001_18000wo1 (n = 6000  | image (n = 6000)): file 이름에 'wo1'이 헷갈릴 수 있는데 error가 난 1개의 폴더도 포함되어있다. 'wo1'은 에러난 1개 폴더안에 최종 파일인 "....brain_mni152_1mm.nii.gz" 에러가 나서 안생겼다는 뜻
                                                                                                           그 이전 단계인 mri폴더 안에 brain.mgz도 잘 안생김
            1 033_S_1016_2016-11-01_S51667_I1051043 not complete

      /ibic/scratch/royseo_workingdir/normalized2mni152_1mm_18001_28000: 1/9 1:45분 현재 9% 돌아감


  - 4. fnirt한 결과
      /ibic/scratch/royseo_workingdir/normalized2mni152_1mm_1-10                (n = 10    | image (n = 10)) 
      /ibic/scratch/royseo_workingdir/normalized2mni152_1mm_11-6000_fnirt_wo11  (n = 5991  | image (n = 5990) + errorlog (n = 1)): 'wo11' 관한 설명 위 flirt 참조
            1  016_S_0702_2013-05-09_S18903_I377040 not complete
            2  021_S_0337_2012-04-24_S14825_I327057 not complete
            3  021_S_0337_2012-04-24_S14825_I327059 not complete
            4  021_S_0337_2012-04-24_S14825_I327062 not complete
            5  021_S_0337_2012-04-24_S14825_I327074 not complete
            6  021_S_0337_2013-05-07_S18878_I389140 not complete
            7  021_S_0337_2013-05-07_S18879_I389138 not complete
            8  021_S_0337_2014-04-24_S21696_I432139 not complete
            9  021_S_0337_2014-04-24_S21696_I432140 not complete
            10 021_S_0337_2015-05-07_S25882_I510022 not complete
            11 021_S_0337_2015-05-07_S25882_I510051 not complete

      /ibic/scratch/royseo_workingdir/normalized2mni152_1mm_6001-12000_wo12 (n = 6000  | image (n = 6000)): 'wo12' 관한 설명 위 flirt 참조
            1  021_S_0984_2014-12-09_S24277_I467254 not complete
            2  023_S_0625_2007-07-16_S35036_I73330 not complete
            3  023_S_0625_2008-01-11_S44497_I90886 not complete
            4  023_S_0625_2008-01-11_S44498_I90882 not complete
            5  023_S_0625_2008-09-29_S56786_I124075 not complete
            6  023_S_0625_2008-09-29_S56787_I124080 not complete
            7  027_S_0644_2012-07-02_S15633_I334521 not complete
            8  027_S_4964_2012-10-06_S17005_I358248 not complete
            9  031_S_0294_2007-09-25_S40106_I79604 not complete
            10 031_S_0618_2006-06-06_S15271_I67110 not complete
            11 031_S_0618_2007-07-02_S36607_I72188 not complete
            12 033_S_0513_2007-05-31_S33069_I67706 not complete

      /ibic/scratch/royseo_workingdir/normalized2mni152_1mm_12001-18000wo1/ (n = 6000): 'wo1' 관한 설명 위 flirt 참조
            1 033_S_1016_2016-11-01_S51667_I1051043 not complete
      
    15. 1-12-2025
    - 1-10-2025일에 6-1-copy_fnirt_warpedfiled.sh를 약간 수정하고 6-2-copy_flirt_brain_mni152_1mm.sh라는 스크립트로 flirt을 첫 18000subjects out of 28002를 두개의 폴더로 옮겼음
    - normalization에서 flirt이 fnirt보다 앞서지만 copy를 시작한 파일이 fnirt파일이라 6
      - 1_linear_1-6000
      - 2_linear_6001-8000
          
 







