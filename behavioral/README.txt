# behavvioral README.txt note
# 1-5-2025 Roy Seo Korea

1. 1-5-2025
- 'ADNI_1_2_3_4_11_14_24_1_02_2025.csv'
- 전에 다운받았던 28002의 이미지에 대한 demographics를 다운 받을 수 없을 것 같아서
- 비슷한 search criteria로 다시 검색했더니 28068개의 이미지가 골라졌다. 그래서 그들의 demographics가 'ADNI_1_2_3_4_11/24behav28068_12_21_24'에 저장되어있음
- 그러나 전에 다운받았던 28002의 이미지의 demographics를 다운 받 을 수 있었다. 1월 2일에 다운 받음


- 4-match_adni_subjects.py
- 'ADNI_1_2_3_4_11_14_24_1_02_2025.csv'이 파일을 은솔 교수님 학생에게 몇가지 작업을 덜어주기 위해
- Adding the full image filenames as the first column by matching Image Data IDs 
   from 'subj_list_28001_raw_prac.log'
- Reordering the resulting dataset to match the order of subjects in the log file : 이것은 내가 recon-all 하고 나서 그때그때 보내기 싶게하기위해서 
  즉, 오늘 6001-12000까지 즉 6000개의 subjects가 recon-all 을 마쳤고 rsync 할 수 있게 되었다면 6001-12000 subjects 개 .log file 순서와 같은 순서로 adni_1234_28002_df.csv 에도 있다.
- Creating a new CSV file '4-adni_1234_28002_dx_age_sex_acqdate.csv' with the combined data

2. 1-13-2025
- All_Subjects_DXSUM_07Jan2025.csv
    - "ADNI1234_subj_n1189.txt" 과 함께 
      '6-filtered_subjects_DXSUM_01_08-2025.py'에 input file로 쓰임
    - 'PTID' = subjID 와 'EXAMDATE' 과 'DIAGNOSIS' = 방문시 받은 진단' 이 포함된 파일
    - 'PTID', 
      'EXAMDATE',
      'DIAGNOSIS
      이 세가지 칼럼을 이용해서 '6-filtered_subjects_DXSUM_01_08-2025.py' 에서
        1189명의 subjects를 여기서  'All_Subjects_DXSUM_07Jan2025.csv' 골라낸다.
        몇개가 골라졌는지 출력
        필터링 테이터에서 누락된 피험자 ID가 있는지 출력
        진단코드가 변한 사람이 있는지 출력
    - 각종 칼럼의 뜻은  'DATADIC_07Jan2025.csv'을 참고 할것
    - 1/7/2025 (ADNI가 있는 미국 시간으로) 에 다운 받음
    - 다운받은 장소
      ADNI에 로그인후 뜨는 페이지에서 (https://ida.loni.usc.edu/home/projectPage.jsp?project=ADNI)
      'Search and Download' -> 'ARC Builder' -> 'Downloads'tab -> 'Study Files' -> 왼쪽 'Assessments' dropdown menu
      -> 'Diagnosis' -> 둘중에 하나임 
                        'Diagnostic Summary - Baseline Changes [ADNI 1, GO, 2, 3, 4]', or 
                        'Diagnostic Summary [ADNI1, GO, 2, 3, 4]' 이거일 가능성이 더 큼
      -> 'Downloads' tab에 들어가보면 내가 받은 다운 받고 싶은 table에 File Name Prefix 가 All Subjects로 되어 있다. 
         

- ADNIMERGE_08Jan2025.csv
    - '7-adding-DX_01_08_2025.py' 에 input으로 쓰임
    - ADNI1, 2, 3, 4, GO가 Merge 된 파일 
    - 1/8/2025 5:20AM(한국시간으로) seoroy15@gmail.com 참조:
      
      "(ADNIMERGE - Key ADNI tables merged into one table [ADNI1,GO,2,3,4]) 
      has combined all the diagnosis information from the various variables 
      in the Diagnostic Summary file to give the participant's diagnosis at each visit, 
      which might also be helpful"

    - 1/8/2025 (ADNI가 있는 미국 시간으로) 에 다운 받음
    - '4-adni_1234_28002_dx_age_sex_acqdate.csv'와 'ADNIMERGE_08Jan2025.csv'를 사용해서 
      '7-add-DX-0108-25.py'파일로
    - 'Image Data ID' (from '4-adni_1234_28002_dx_age_sex_acqdate.csv') with 'IMAGEUID' (from 'ADNIMERGE_08Jan2025.csv')
      'Subject' (from '4-adni_1234_28002_dx_age_sex_acqdate.csv') with PTID (from 'ADNIMERGE_08Jan2025.csv') 로 매치해서
      'DX'칼럼을 채워줌
    - 그다음 '8-DX-filled_1-10-25.py' Script로 'Acq Date', 'Subject'가 일치 하는 한 'DX_fill' 칼럼에 같은 진단명으로 채움
    - 다운 받은 장소
      ADNI에 로그인후 뜨는 페이지에서 (https://ida.loni.usc.edu/home/projectPage.jsp?project=ADNI)
      'Search and Download' -> 'Study Files' -> 'Study Info' -> 'ADNIMERGE - Key ADNI tables merged into one table [ADNI1,GO,2,3]'
    

- DATADIC_07Jan2025.csv
 - Data dictionary 임
 - 다운 받은 장소 
   ADNI에 로그인후 뜨는 페이지에서 (https://ida.loni.usc.edu/home/projectPage.jsp?project=ADNI)
      'Search and Download' -> 'ARC Builder' -> 'Study Info' -> 'Data & Database' 테이블에 있음 - 'Data Dictionary [ADNI1,GO,2,3,4]'
    
