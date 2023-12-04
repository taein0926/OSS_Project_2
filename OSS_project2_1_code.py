import pandas as pd

# 각 연도별 H, avg, HR, OBP에 따라 상위 10명의 선수를 출력하는 함수
def print_top10(data_df, year, stat):
    year_data = data_df[data_df['year'] == year]
    sorted_data = year_data.sort_values(by=stat, ascending=False)
    top10 = sorted_data.head(10)
    print(f"{year}년도 {stat} 기준 상위 10명 선수:")
    print(top10[['batter_name', stat]])
    print()

# 각 포지션별로 2018년도 승리 기여도가 가장 높은 선수를 출력하는 함수
def highest_war_by_position(data_df, year):
    positions = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
    year_data = data_df[data_df['year'] == year]
    print(f"{year}년도 각 포지션별 승리 기여도가 가장 높은 선수:")
    for position in positions:
        position_data = year_data[year_data['cp'] == position]
        highest_war_player = position_data.loc[position_data['war'].idxmax()]
        print(f"{position}: {highest_war_player['batter_name']} (WAR: {highest_war_player['war']:.3f})")
    print()

# 연봉과 가장 높은 상관관계를 가진 통계치를 출력하는 함수
def highest_correlation_with_salary(data_df):
    stats = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']
    correlations = data_df[stats + ['salary']].corr()['salary'].drop('salary')
    highest_correlation_stat = correlations.idxmax()
    print(f"연봉과 가장 높은 상관관계를 가진 통계치: {highest_correlation_stat} (상관계수: {correlations[highest_correlation_stat]:.3f})")
    print()

if __name__ == '__main__':
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    for year in range(2015, 2019):
        for stat in ['H', 'avg', 'HR', 'OBP']:
            print_top10(data_df, year, stat)

    highest_war_by_position(data_df, 2018)

    highest_correlation_with_salary(data_df)
