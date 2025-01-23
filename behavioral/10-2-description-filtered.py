'''
01-23-2025 Roy Seo Korea
    주어진 CSV 파일에서 Description 열에 MPR; GradWarp; B1 Correction; N3 값이 포함된 행만 남기고, 
    다른 모든 행은 제거하여 결과를 새로운 CSV 파일로 저장하는 Python 스크립트

    (MPR; GradWarp; B1 Correction; N3여기까지 처리된것을 사용하기로 결정했는데 이유는 김박사님과 대화를 통해서 scaled, scaled2번한 것이 크게 dramatic하게 변화를 주진 않았을것 같다는 생각이 들었고, 
     또 10-1-description-stat.py 결과를 보면 적당하게 처리되면서 데이타의 수도 많은 것이 MPR; GradWarp; B1 Correction; N3 여기까지 처리 된것임)

csv file: 8-2-DX_filled2.csv
outputfile: 10-2-description-filtered.csv
'''

import pandas as pd

# Read CSV file
df = pd.read_csv('8-2-DX_filled2.csv')

# Filter rows containing specific Description
filtered_df = df[df['Description'] == 'MPR; GradWarp; B1 Correction; N3']

# Save filtered dataframe to new CSV
filtered_df.to_csv('10-2-description-filtered.csv', index=False)