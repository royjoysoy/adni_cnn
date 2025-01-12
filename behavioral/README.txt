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

2. 1-12-2025
- 