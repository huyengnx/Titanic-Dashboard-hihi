import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils
#한글
import matplotlib.font_manager as fm

st.set_page_config(page_title="타이타닉 생존자 대시보드", layout="wide")
st.image("./img/titanic.jpg", caption="타이타닉 - 재난에서 배우는 머신러닝", use_container_width=True)
# column은 container로 바꿀 수도 있다.   

fm.fontManager.addfont('./font/NanumGothic-Regular.ttf')
fm.fontManager.addfont('./font/NanumGothic-Bold.ttf')
fm.fontManager.addfont('./font/NanumGothic-ExtraBold.ttf')

# Set font family for matplotlib
plt.rcParams['font.family'] = 'Nanum Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Set font for Streamlit (CSS)
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Nanum Gothic', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)


# Load datas
@st.cache_data
def load_data():
    train_df = pd.read_csv('data/train.csv')
    return train_df

def main():
    st.title("🚢 타이타닉 생존자 대시보드")
    
    # Menu (selectbox)
    page = st.sidebar.selectbox("메뉴", ["홈", "생존 개요", "연령별 생존", "좌석 등급별 생존", 'Data Source'])

    # Load data
    df = load_data()

#----------------------------------------------------------------------  
    # Page1: Home
    if page == "홈":
        st.title("데이터 출처: Kaggle")
        st.markdown('Titanic - Machine Learning from Disaster')
#------------------------------------------------------------------------
    # Page2: Survival Overview (생존개요)
    elif page == "생존 개요":
        st.subheader("생존자 수 & 생존율")

    # 추가: 총 생존자 / 사망자 수 표시
        total_counts = df['Survived'].value_counts().sort_index()
        not_survived_count = total_counts[0]
        survived_count = total_counts[1]
        
        st.markdown(f"🟥 **총 사망자 수:** {not_survived_count}명")
        st.markdown(f"🟩 **총 생존자 수:** {survived_count}명")
        
    # 생존자의 수수
    # Grouped Bar Chart: 생존자 수 + 성별
        survival_gender_counts = df.groupby(['Survived', 'Sex']).size().unstack(fill_value=0)

        # 생존자의 수수
        # Grouped Bar Chart: 생존자 수 + 성별
        survival_gender_counts = df.groupby(['Survived', 'Sex']).size().unstack(fill_value=0)
        fig, ax1 = plt.subplots(figsize=(8,5))
        bar_width = 0.35
        x = range(len(survival_gender_counts.index))
        # Bar Male
        bars1 = ax1.bar([i - bar_width/2 for i in x], survival_gender_counts['male'], 
                    width=bar_width, color='lightblue', label='남성')
        # Bar Female
        bars2 = ax1.bar([i + bar_width/2 for i in x], survival_gender_counts['female'], 
                    width=bar_width, color='lightcoral', label='여성')   
        ax1.set_xlabel('생존 여부')
        ax1.set_ylabel('탑승객 수')
        ax1.set_title('성별 생존자 수 (그룹형 막대 그래프)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['사망', '생존'])
        ax1.legend(title='성별')

        for bar in bars1 + bars2:
            height = bar.get_height()
            ax1.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0,3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=12 )
        # Survival Rate by Gender - Pie Chart
        survival_rate_gender = df.groupby('Sex')['Survived'].mean().reindex(['female', 'male']).fillna(0)
        fig_survival_rate, ax2 = plt.subplots(figsize=(2.7,2.7))
        
        wedges, texts, autotexts = ax2.pie(
                survival_rate_gender.values, 
                labels=None, 
                autopct='%1.1f%%', 
                colors=['lightcoral', 'lightblue' ], 
                startangle=90)
        
        ax2.legend(
            wedges, 
            ['여성', '남성'], #<--: Legend
            title='성별',
            loc='center left', 
            bbox_to_anchor=(1, 0.5)
            )
        
        ax2.set_title('성별 생존율 (원형 차트)')

        # Layout of 2 columns
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig)
        with col2:
            st.pyplot(fig_survival_rate)        
# -----------------------------------------------    
    # Page3: Age-based Survival (연령별 생존)
    elif page == "연령별 생존":
        st.subheader("연령대별 생존자 수 (라인 차트)")

        # 총 사망자 / 생존자 수 표시
        total_counts = df['Survived'].value_counts().sort_index()
        not_survived_count = total_counts[0]
        survived_count = total_counts[1]

        st.markdown(f"🟥 **총 사망자 수:** {not_survived_count}명")
        st.markdown(f"🟩 **총 생존자 수:** {survived_count}명")

        # 슬라이더 추가 → 나이 범위 선택
        min_age = int(df['Age'].min())
        max_age = int(df['Age'].max())

        selected_age_range = st.slider(
            "나이 범위를 선택하세요:",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age)
        )

        # 선택한 나이 범위에 따라 데이터 필터링
        df_age_filtered = df[(df['Age'] >= selected_age_range[0]) & (df['Age'] <= selected_age_range[1])].copy()
        df_age_filtered = df_age_filtered.dropna(subset=['Age'])

        # Age bin 생성
        df_age_filtered['Age_bin'] = pd.cut(df_age_filtered['Age'], bins=20)

        # Groupby Age_bin + Survived → count
        age_survival_counts = df_age_filtered.groupby(['Age_bin', 'Survived']).size().reset_index(name='Count')

        # Age_bin midpoint 계산 → line chart X축에 쓰기
        age_survival_counts['Age_bin_mid'] = age_survival_counts['Age_bin'].apply(lambda x: x.left + (x.right - x.left)/2)

        # 색상 설정
        color_survived = "#2257e9"       # 생존
        color_not_survived = "#d46161"   # 사망

        # Line chart 그리기
        fig, ax = plt.subplots(figsize=(10,6))

        sns.lineplot(
            data=age_survival_counts[age_survival_counts['Survived'] == 0],
            x='Age_bin_mid',
            y='Count',
            color=color_not_survived,
            linewidth=2.5,
            marker='o',
            label='사망',
            ax=ax
        )
        sns.lineplot(
            data=age_survival_counts[age_survival_counts['Survived'] == 1],
            x='Age_bin_mid',
            y='Count',
            color=color_survived,
            linewidth=2.5,
            marker='o',
            label='생존',
            ax=ax
        )
        ax.set_xlabel('나이')
        ax.set_ylabel('탑승객 수')
        ax.set_title('연령대별 생존자 수 (라인 차트)')
        ax.legend(title='생존 여부')

        st.pyplot(fig)
#--------------------------------------------------------------------------
    # Page4: Survival by Ticket Class (좌석 등급별 생존)
    elif page == "좌석 등급별 생존":
        st.subheader("좌석 등급별 생존자 수 (그룹형 막대 그래프)")

        pclass_survival_counts = df.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
        pclass_survival_counts.columns = ['사망', '생존']

        fig, ax = plt.subplots(figsize=(8,5))
        bar_width = 0.35
        x = range(len(pclass_survival_counts.index))
        # Bar Not Survived (사망)
        bars1 = ax.bar([i - bar_width/2 for i in x], pclass_survival_counts['사망'],
                    width=bar_width, color='salmon', label='사망')
        # Bar Survived (생존)
        bars2 = ax.bar([i + bar_width/2 for i in x], pclass_survival_counts['생존'],
                    width=bar_width, color='mediumseagreen', label='생존')

        ax.set_xlabel('좌석 등급 (Pclass)')
        ax.set_ylabel('탑승객 수')
        ax.set_title('좌석 등급별 생존자 수 (그룹형 막대 그래프)')
        ax.set_xticks(x)
        ax.set_xticklabels(pclass_survival_counts.index.astype(str))
        ax.legend(title='생존 여부')
       
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=12)

        st.pyplot(fig)
#-----------------------------------------------------------------
    # Page5: Data Source
    elif page == "Data Source":
        st.subheader("📊 Data Source - 전체 데이터 보기")

        # 각 CSV 파일 불러오기
        try:
            train_df = pd.read_csv('data/train.csv')
            st.markdown("### **train.csv**")
            st.dataframe(train_df)
        except Exception as e:
            st.error(f"Train CSV 불러오기 오류: {e}")

        try:
            test_df = pd.read_csv('data/test.csv')
            st.markdown("###  **test.csv**")
            st.dataframe(test_df)
        except Exception as e:
            st.error(f"Test CSV 불러오기 오류: {e}")

        try:
            gender_df = pd.read_csv('data/gender_submission.csv')
            st.markdown("### **Gender Data**")
            st.dataframe(gender_df)
        except Exception as e:
            st.error(f"gender_submission.csv 불러오기 오류: {e}")


if __name__ == "__main__":
    main()
