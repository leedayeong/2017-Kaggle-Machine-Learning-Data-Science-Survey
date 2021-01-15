#!/usr/bin/env python
# coding: utf-8

# 참고 및 출처 URL:
# 
# *   [Data Science FAQ | Kaggle](https://www.kaggle.com/rounakbanik/data-science-faq)
# 
# *   [Novice to Grandmaster | Kaggle](https://www.kaggle.com/ash316/novice-to-grandmaster)
# *   [박조은 강사님 Github](https://github.com/corazzon/KaggleStruggle/blob/master/kaggle-survey-2017/Kaggle-ML-DS-survey-2017-EDA-FAQ.ipynb)
# 
# 
# 이 설문조사의 결과를 바탕으로 데이터사이언스와 머신러닝과 관련 된 인사이트를 얻어볼 수 있지 않을까 가설을 세워본다.

# # 캐글러를 대상으로 한 설문조사
# 
# *   설문기간 : 2017년 8월 7일부터 8월 25일까지
# *   평균 응답 시간은 16.4 분
# *   171 개 국가 및 지역에서 16,716 명의 응답자
# *   특정 국가 또는 지역에서 응답자가 50 명 미만인 경우 익명을 위해 그룹을 '기타'그룹으로 그룹화
# *   설문 조사 시스템에 신고 된 응답자를 스팸으로 분류하거나 취업 상태에 관한 질문에 답변하지 않은 응답자는 제외(이 질문은 첫 번째 필수 질문이기에 응답하지 않으면 응답자가 다섯 번째 질문 이후 진행되지 않음)
# *   대부분의 응답자는 이메일 목록, 토론 포럼 및 소셜 미디어 Kaggle 채널을 통해 설문을 알게 됨
# *   급여데이터는 일부 통화에 대해서만 받고 해당 되는 통화에 기준하여 작성하도록 함
# *   미국 달러로 급여를 계산할 수 있도록 USD로 환산 한 csv를 제공
# *   질문은 선택적
# *   모든 질문이 모든 응답자에게 보여지는 것은 아님
# *   취업을 한 사람과 학생을 나누어 다른 질문을 함
# *   응답자의 신원을 보호하기 위해 주관식과 객관식 파일로 분리
# *   객관식과 자유 형식 응답을 맞추기 위한 키를 제공하지 않음
# *   주관식 응답은 같은 행에 나타나는 응답이 반드시 동일한 설문 조사자가 제공하지 않도록 열 단위로 무작위 지정

# # 데이터 파일

# 5 개의 데이터 파일을 제공
# 
# 
# 
# *   schema.csv : 설문 스키마가있는 CSV 파일입니다. 이 스키마에는 multipleChoiceResponses.csv 및 freeformResponses.csv의 각 열 이름에 해당하는 질문이 포함되어 있습니다.
# *  multipleChoiceResponses.csv : 객관식 및 순위 질문에 대한 응답자의 답변, 각 행이 한 응답자의 응답
# *   freeformResponses.csv : Kaggle의 설문 조사 질문에 대한 응답자의 주관식 답변입니다. 임의로 지정되어 각 행이 같은 응답자를 나타내지 않음
# *   conversionRates.csv : R 패키지 "quantmod"에서 2017 년 9 월 14 일에 액세스 한 통화 변환율 (USD)
# *   RespondentTypeREADME.txt : schema.csv 파일의 "Asked"열에 응답을 디코딩하는 스키마입니다.

# In[1]:


# 노트북 안에서 그래프를 그리기 위해
get_ipython().run_line_magic('matplotlib', 'inline')

#import the standard Python Scientific Libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

#Suppress Deprecation and Incorrect Usage Warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


question = pd.read_csv('schema.csv')
question.shape


# In[3]:


question.head()


# In[4]:


#판다스로 선다형 객관식 문제에 대한 응답을 가져 옴
mcq = pd.read_csv('multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory=False)
mcq.shape


# In[5]:


mcq.head(10)


# In[6]:


# missingno는 NaN 데이터들에 대해 시각화를 해준다.
# NaN 데이터의 컬럼이 많아 아래 그래프만으로는 내용을 파악하기 어렵다.
import missingno as msno

msno.matrix(mcq, figsize=(12,5))


# * 16,716 명의 데이터와 228개의 선다형 객관식문제와 62개의 주관식 질문에 대한 응답이다. (총 290개의 질문) 응답하지 않은 질문이 많음
# 

# # 설문통계

# In[7]:


#성별
sns.countplot(y='GenderSelect', data=mcq)


# 여성보다는 남성의 비율이 훨씬 높은 편이다.

# In[8]:


#국가별 응답수
con_df = pd.DataFrame(mcq['Country'].value_counts())
#print(con_df)
#'country'컬럼을 인덱스로 지정해 주고
con_df['국가']=con_df.index
#컬럼의 순서대로 응답 수, 국가로 컬럼명을 지정해 줌
con_df.colums = ['응답 수', '국가']
#index 컬럼을 삭제하고 순위를 알기위해 reset_index()를 해줌
#우리 나라는 18위, 전체 52개국에서 참여했지만 20위까지만 본다.
con_df=con_df.reset_index().drop('index', axis=1)
con_df.head(20)


# In[9]:


# 연령에 대한 정보를 본다.
mcq['Age'].describe()


# In[10]:


sns.distplot(mcq[mcq['Age']>0]['Age'])


# 응답자의 대부분이 어리며, 20대부터 급격히 늘어나며, 30대가 가장 많다. 평균 나이는 32세다.

# # 학력

# In[11]:


sns.countplot(y='FormalEducation', data=mcq)


# 학사 학위를 가진 사람보다 석사 학위를 가지고 있는 사람이 많으며, 박사학위를 가지고 있는 사람들도 많다.

# In[12]:


pd.DataFrame(mcq['MajorSelect'].value_counts())


# # 전공

# In[13]:


# value_counts 를 사용하면 그룹화 된 데이터의 카운트 값을 보여준다. 
# normalize=True 옵션을 사용하면, 
# 해당 데이터가 전체 데이터에서 어느정도의 비율을 차지하는지 알 수 있다.
mcq_major_count = pd.DataFrame(mcq['MajorSelect'].value_counts())
mcq_major_percent = pd.DataFrame(mcq['MajorSelect'].value_counts(normalize=True))
mcq_major_df = mcq_major_count.merge(mcq_major_percent, left_index=True, right_index=True)
mcq_major_df.columns = ['응답 수', '비율']
mcq_major_df


# 컴퓨터 전공자들이 33%로 가장 많으며, 다음으로 수학, 공학, 전기 공학 순이다.

# In[14]:


#사람들의 전공 현황
plt.figure(figsize=(6,8))
sns.countplot(y='MajorSelect', data = mcq)


# # 취업 여부 

# In[15]:


mcq_es_count = pd.DataFrame(mcq['EmploymentStatus'].value_counts())
mcq_es_percent = pd.DataFrame(mcq['EmploymentStatus'].value_counts(normalize=True))
mcq_es_df = mcq_es_count.merge(mcq_es_percent, left_index=True, right_index=True)
mcq_es_df.columns = ['응답 수', '비율']
mcq_es_df


# In[16]:


sns.countplot(y='EmploymentStatus', data=mcq)


# 응답자의 대부분이 65%가 풀타임으로 일하고 있으며, 그 다음으로 구직자가 12%다.

# #  프로그래밍 경험

# * 'Tenure'항목은 데이터사이언스 분야에서 코딩 경험이 얼마나 되는지에 대한 질문이다. 대부분이 5년 미만이며, 특히 1~2년의 경험을 가진 사람들이 많다.

# In[17]:


sns.countplot(y='Tenure', data=mcq)


# In[18]:


korea = mcq.loc[(mcq['Country']=='South Korea')]

print('The number of interviewees in Korea: ' + str(korea.shape[0]))

sns.distplot(korea['Age'].dropna())
plt.title('Korean')
plt.show()


# In[19]:


pd.DataFrame(korea['GenderSelect'].value_counts())


# In[20]:


sns.countplot(x='GenderSelect', data=korea)
plt.title('Korean')


# In[21]:


figure, (ax1, ax2) = plt.subplots(ncols=2)
figure.set_size_inches(12,5)

sns.distplot( korea['Age'].loc[korea['GenderSelect']=='Female'].dropna(),
             norm_hist=False, color=sns.color_palette("Paired")[4], ax=ax1)
ax1.set_title('Korean Female')

sns.distplot(korea['Age'].loc[korea['GenderSelect']=='Male'].dropna(), 
             norm_hist=False, color=sns.color_palette("Paired")[0], ax=ax2)
ax2.set_title('Korean Male')


# In[22]:


sns.barplot(x=korea['EmploymentStatus'].unique(),y=korea['EmploymentStatus'].value_counts()/len(korea))
plt.xticks(rotation=30, ha='right')
plt.title('Employment status of the korean')
plt.ylabel('')
plt.show()


# In[23]:


korea['StudentStatus'] = korea['StudentStatus'].fillna('No')
sns.countplot(x='StudentStatus', data=korea)
plt.title('korean')
plt.show()


# In[24]:


full_time = mcq.loc[(mcq['EmploymentStatus'] == 'Employed full-time')]
print(full_time.shape)
looking_for_job = mcq.loc[(
    mcq['EmploymentStatus'] == 'Not employed, but looking for work')]
print(looking_for_job.shape)


# # 자주 묻는 질문 FAQ 

# * 초보자들이 묻는 가장 일반적인 질문에 대한 답을 시각화 해본다.

# ## Q1. Python과 R중 어떤 언어를 배워야 할까요?
# 

# In[25]:


sns.countplot(y='LanguageRecommendationSelect', data=mcq)


# 파이썬을 명확하게 선호하고 있는 것으로 보여지며, 전문가와 강사들이 선호하는 언어를 알아본다.

# In[26]:


# 현재 하고 있는 일
sns.countplot(y=mcq['CurrentJobTitleSelect'])


# In[27]:


# 현재 하고 있는 일에 대한 전체 응답수
mcq[mcq['CurrentJobTitleSelect'].notnull()]['CurrentJobTitleSelect'].shape


# In[28]:


# 현재 하고 있는 일에 대한 응답을 해준 사람 중 Python과 R을 사용하는 사람
# 응답자들이 실제 업무에서 어떤 언어를 주로 사용하는지 볼 수 있다.
data = mcq[(mcq['CurrentJobTitleSelect'].notnull()) & (
    (mcq['LanguageRecommendationSelect'] == 'Python') | (
        mcq['LanguageRecommendationSelect'] == 'R'))]
print(data.shape)
plt.figure(figsize=(8, 10))
sns.countplot(y='CurrentJobTitleSelect', 
              hue='LanguageRecommendationSelect', 
              data=data)


# 데이터사이언티스트들은 Python을 주로 사용하지만 R을 사용하는 사람들도 제법 된다. 하지만 소프트웨어 개발자들은 Python을 훨씬 더 많이 사용하며, Python보다 R을 더 많이 사용하는 직업군은 통계 학자들이다.

# ## Q2. 데이터 사이언스 분야에서 앞으로 크게 주목받을 것은 무엇일까요?

# * 관련 분야의 종사자가 아니더라도 빅데이터, 딥러닝, 뉴럴네트워크 같은 용어에 대해 알고 있다. 응답자들이 내년에 가장 흥미로운 기술이 될 것이라 응답한 것이다.

# ### 데이터사이언스 툴

# In[29]:


mcq_ml_tool_count = pd.DataFrame(
    mcq['MLToolNextYearSelect'].value_counts())
mcq_ml_tool_percent = pd.DataFrame(
    mcq['MLToolNextYearSelect'].value_counts(normalize=True))

mcq_ml_tool_df = mcq_ml_tool_count.merge(
    mcq_ml_tool_percent, left_index=True, right_index=True).head(20)
mcq_ml_tool_df.columns = ['응답 수', '비율']
mcq_ml_tool_df


# In[30]:


data = mcq['MLToolNextYearSelect'].value_counts().head(20)
sns.barplot(y=data.index, x=data)


# ###  다음 해에 주목할 만한 Data Science Methods

# In[31]:


data = mcq['MLMethodNextYearSelect'].value_counts().head(15)
sns.barplot(y=data.index, x=data)


# 응답에 대한 통계를 보면 딥러닝과 뉴럴넷이 엄청나게 인기가 있을 것이고 시계열 분석, 베이지안, 텍스트 마이닝 등의 내용이 있다. 중간 쯤에 부스팅과 배깅 같은 앙상블 메소드도 있다.

# ## Q3. 어디에서 데이터 사이언스를 배워야 할까요?

# In[32]:


mcq['LearningPlatformSelect']


# In[33]:


# 한 줄로 되어있는 여러개의 응답을 각각의 원소로 나눠서 리스트에 저장함
# 즉 Kaggle, Online courses -> [Kaggle, Online courses]

# 1. mcq의 컬럼 LearningPlatformSelect에 존재하는 값을 string값으로 바꿔줌
#   string 값으로 바꿔주는 이유 : 뒤에 나오는 lambda가 split(',')를 수행하는데 이때 string인 경우에만 사용가능하기 때문이다.
#   NaN의 경우 이 때 데이터 타입은 float 로 인식 되기 때문에 astype('str')을 해주는 것임

mcq['LearningPlatformSelect'] = mcq['LearningPlatformSelect'].astype('str').apply(lambda x: x.split(','))
mcq['LearningPlatformSelect']


# In[34]:


# 리스트의 원소 값을 분리해서 한 행에 하나의 데이터만 들어가도록 처리함
# 즉 [ Kaggle, Online courses] -> [Kaggle]
#                              -> [Online courses]

# 1-1. mcq의 LearningPlatformSelect컬럼 중, axis가 1인 즉, 열의 값을 선택해서 내부의 데이터를 Series 데이터 구조로 바꾼다.
#      mcq.apply(lambda x: pd.Series(x['LearningPlatformSelect']), axis=1)

s = mcq.apply(
    lambda x: pd.Series(x['LearningPlatformSelect']),
    axis=1)
print(s)


# In[35]:


# 1-2. 리스트로 표횐된 자료를 .stack()을 이용해 2level 데이터 프레임으로 바꿔줌 
#                 (2level 이란 인덱스 2개사용해 2차원 배열 구조를 시각화 한 것)
#      .stack() 메서드 사용하면 0번 인덱스에 저장되어있던 값이 세부항목을 리스트가 아닌 리스트에 달려있던 인덱스 넘버로 출력됨
s = mcq.apply(
    lambda x: pd.Series(x['LearningPlatformSelect']),
    axis=1).stack()
s.head(10)


# In[36]:


# 1-3. 2level 인덱스 중, 하나를 없애서 보기 편하게 바꿈
#     .stack()까지 2레벨의 데이터 프레임 인덱스 중, level=1 의 인덱스를 drop함
#      즉, 위에서 0 1 2 3 0 0... 로 진행되는 인덱스 drop시킴
#      참고) 맨 앞의 0     12  처럼 표현되는 인덱스 level은 0 임

s = mcq.apply(
    lambda x: pd.Series(x['LearningPlatformSelect']),
    axis=1).stack().reset_index(level=1, drop=True)
# <class 'pandas.core.series.Series'> 인 s에 컬럼 이름을 platform이라고 명명해줌
s.name = 'platform'
s.head(10)


# In[37]:


# 출력 될 figure 크기 설정 .figsize(가로, 세로)
plt.figure(figsize=(6,8))

# s에서 s가 nan인 값을 제외하고 각 원소가 몇개 있는지 갯수를 셈
# 그 후 상위 15개 선택함
data = s[s != 'nan'].value_counts().head(15)
sns.barplot(y=data.index, x=data)


# * Kaggle은 우리 응답자들 사이에서 가장 인기있는 학습 플랫폼
# * 그러나 이 설문 조사를 실시한 곳이 Kaggle이기 때문에 응답이 편향되었을 수 있음
# * 온라인 코스, 스택 오버플로 및 유튜브 (YouTube) 상위 5 대 최우수 학습 플랫폼은 대학 학위나 교과서의 중요도보다 높다.

# In[38]:


# 설문내용과 누구에게 물어봤는지를 찾아봄

# question은 schema.csv를 담고 있는 데이터 프레임
# question의 Column이라는 명칭을 가진 컬럼에서 LearningCategory라는 글자가 들어간 컬럼만 선택해서 컬럼의 명칭과 데이터를 qc에 담는다.
qc = question.loc[question['Column'].str.contains('LearningCategory')]
print(qc.shape)
qc


# In[39]:


# mcq의 컬럼중 LearningPlatformUsefulness가 들어간 것들을 찾아서 use_features 리스트에 넣음

use_features = [x for x in mcq.columns if x.find(
    'LearningPlatformUsefulness') != -1]


# In[40]:


# 학습플랫폼과 유용함에 대한 연관성을 살펴본다.

# fdf : 데이터를 담을 딕셔너리 타입의 객체 생성 
fdf = {}
for feature in use_features:
    a = mcq[feature].value_counts()
    a = a/a.sum()
    # fdf에 LearningPlatformUsefulnessArxiv의 LearningPlatformUsefulness을 뺀 것을 key로, a를 value로 한다.
    fdf[feature[len('LearningPlatformUsefulness'):]] = a
# fdf 데이터 행(Very useful, Somewhat useful, Not Useful), 열(백분율 즉, a값) 
print(fdf)


# In[41]:


# Very useful 에 대한 내림차순 정렬 수행
# 내림차순 정렬 위해 fdf의 행과열 위치 바꿔야함 (transpose)

fdf = pd.DataFrame(fdf).transpose().sort_values(
    'Very useful', ascending=False)


# In[42]:


# 학습플랫폼들이 얼마나 유용한지에 대한 상관관계를 그려본다.
# annot=True 이면 각 셀에 데이터 값을 씀, annot 은 annotation(주석)의 약자

plt.figure(figsize=(10,10))
sns.heatmap(
    fdf.sort_values(
        "Very useful", ascending=False), annot=True)


# In[43]:


# 유용함의 정도를 각 플랫폼별로 그룹화 해서 본다.
fdf.plot(kind='bar', figsize=(20,8),
         title="Usefullness of Learning Platforms")
plt.xticks(rotation=60, ha='right')


# 실제로 프로젝트를 해보는 것에 대해 74.7%의 응답자가 응답했고 매우 유용하다고 표시했다. SO는 스택오버플로우가 아닐까 싶고, 캐글, 수업, 책이 도움이 많이되는 편이다. 팟캐스트는 매우 유용하지 않지만 때때로 유용하다는 응답은 가장 많았다.

# In[44]:


cat_features = [x for x in mcq.columns if x.find(
    'LearningCategory') != -1]
cat_features


# In[45]:


cdf = {}
for feature in cat_features:
    cdf[feature[len('LearningCategory'):]] = mcq[feature].mean()

# 파이차트를 그리기 위해 평균 값을 구해와서 담아준다.
cdf = pd.Series(cdf)
cdf


# In[46]:


# 학습 플랫폼 별 도움이 되는 정도를 그려본다.
plt.pie(cdf, labels=cdf.index, 
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title("Contribution of each Platform to Learning")
plt.show()


# ## Q4. 데이터과학을 위해 높은 사양의 컴퓨터가 필요한가요?

# In[47]:


# 설문내용과 누구에게 물어봤는지를 찾아봄
qc = question.loc[question[
    'Column'].str.contains('HardwarePersonalProjectsSelect')]
print(qc.shape)
qc


# In[48]:


mcq[mcq['HardwarePersonalProjectsSelect'].notnull()][
    'HardwarePersonalProjectsSelect'].shape


# In[49]:


mcq['HardwarePersonalProjectsSelect'
   ] = mcq['HardwarePersonalProjectsSelect'
          ].astype('str').apply(lambda x: x.split(','))
s = mcq.apply(lambda x: 
              pd.Series(x['HardwarePersonalProjectsSelect']),
              axis=1).stack().reset_index(level=1, drop=True)
s.name = 'hardware'


# In[50]:


s = s[s != 'nan']


# In[51]:


pd.DataFrame(s.value_counts())


# 맥북을 사용하는 응답자가 가장많고, 랩탑과 함께 클라우드를 사용하는 사람들이 그 다음이고 적당한 GPU를 가진 게임용 노트북을 사용하는 사례가 그 다음이다.

# ## Q5. 데이터 사이언스 공부에 얼마나 많은 시간을 사용하는지?

# In[52]:


plt.figure(figsize=(6, 8))
sns.countplot(y='TimeSpentStudying', 
              data=mcq, 
              hue='EmploymentStatus'
             ).legend(loc='center left',
                      bbox_to_anchor=(1, 0.5))


# 풀타임으로 일하는 사람들은 2~10시간 일하는 비율이 높으며, 풀타임으로 일하는 사람보다 일을 찾고 있는 사람들이 더 많은 시간을 공부하는 편이다.
# 
# 하지만 응답자 중 대부분이 풀타임으로 일하고 있는 사람들이라는 것을 고려할 필요가 있다.

# In[53]:


full_time = mcq.loc[(mcq['EmploymentStatus'] == 'Employed full-time')]
print(full_time.shape)
looking_for_job = mcq.loc[(
    mcq['EmploymentStatus'] == 'Not employed, but looking for work')]
print(looking_for_job.shape)


# In[54]:


figure, (ax1, ax2) = plt.subplots(ncols=2)

figure.set_size_inches(12,5)
sns.countplot(x='TimeSpentStudying', 
              data=full_time, 
              hue='EmploymentStatus', ax=ax1
             ).legend(loc='center right',
                      bbox_to_anchor=(1, 0.5))

sns.countplot(x='TimeSpentStudying', 
              data=looking_for_job, 
              hue='EmploymentStatus', ax=ax2
             ).legend(loc='center right',
                      bbox_to_anchor=(1, 0.5))


# ## Q6. 블로그, 팟캐스트, 수업 기타 등등 추천할만한 것이 있는지?

# In[55]:


# 판다스로 선다형 객관식 문제에 대한 응답을 가져 옴
mcq = pd.read_csv('multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory=False)
mcq.shape


# In[56]:


mcq['BlogsPodcastsNewslettersSelect'] = mcq[
    'BlogsPodcastsNewslettersSelect'
].astype('str').apply(lambda x: x.split(','))
mcq['BlogsPodcastsNewslettersSelect'].head()


# In[57]:


s = mcq.apply(lambda x: pd.Series(x['BlogsPodcastsNewslettersSelect']),
              axis=1).stack().reset_index(level=1, drop=True)
s.name = 'platforms'
s.head()


# In[58]:


s = s[s != 'nan'].value_counts().head(20)


# In[59]:


plt.figure(figsize=(6,8))
plt.title("Most Popular Blogs and Podcasts")
sns.barplot(y=s.index, x=s)


# KDNuggets Blog, R Bloggers Blog Aggregator 그리고 O'Reilly Data Newsletter 가 가장 유용하다고 투표를 받았다. 데이터 사이언스 되기라는 팟캐스트도 유명한 듯 하다.

# [Machine Learning, Data Science, Big Data, Analytics](https://www.kdnuggets.com/)
# 
# [Becoming a Data Scientist - YouTube - YouTube](https://www.youtube.com/channel/UCfxnrdBM1YRV9j2MB8aiy4Q)
# 
# [Siraj Raval - YouTube - YouTube](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A)

# In[60]:


mcq['CoursePlatformSelect'] = mcq[
    'CoursePlatformSelect'].astype(
    'str').apply(lambda x: x.split(','))
mcq['CoursePlatformSelect'].head()


# In[61]:


t = mcq.apply(lambda x: pd.Series(x['CoursePlatformSelect']),
              axis=1).stack().reset_index(level=1, drop=True)
t.name = 'courses'
t.head(20)


# In[62]:


t = t[t != 'nan'].value_counts()


# In[63]:


plt.title("Most Popular Course Platforms")
sns.barplot(y=t.index, x=t)


# Coursera와 Udacity가 가장 인기있는 플랫폼이다.

# ## Q7. 데이터 사이언스 직무에서 가장 중요하다고 생각되는 스킬은?

# In[64]:


job_features = [
    x for x in mcq.columns if x.find(
        'JobSkillImportance') != -1 
    and x.find('JobSkillImportanceOther') == -1]

job_features


# In[65]:


jdf = {}
for feature in job_features:
    a = mcq[feature].value_counts()
    a = a/a.sum()
    jdf[feature[len('JobSkillImportance'):]] = a

jdf


# In[66]:


jdf = pd.DataFrame(jdf).transpose()
jdf


# In[67]:


plt.figure(figsize=(10,6))
sns.heatmap(jdf.sort_values("Necessary", 
                            ascending=False), annot=True)


# In[68]:


jdf.plot(kind='bar', figsize=(12,6), 
         title="Skill Importance in Data Science Jobs")
plt.xticks(rotation=60, ha='right')


# 꼭 필요한 것으로 Python, R, SQL, 통계, 시각화가 있다.
# 
# 있으면 좋은 것은 빅데이터, 학위, 툴 사용법, 캐글랭킹, 무크가 있다.

# ## Q8. 데이터 과학자의 평균 급여는 얼마나 될까?

# In[69]:


# 데이터 프레임의 컬럼을 최대 12개로 늘린다.

pd.set_option('display.max_columns', 12)

# schema.csv를 question 객체에 담기

question = pd.read_csv('schema.csv')

# multipleChoiceResponses.csv를 mcq에 담기. encoding='ISO-8859-1'은 범용적으로 사용되는 인코딩이다.

mcq = pd.read_csv('multipleChoiceResponses.csv', encoding='ISO-8859-1', low_memory=False)

# 한국 자료를 따로 설정해준다.

korea = mcq.loc[mcq.Country == 'South Korea']


# In[70]:


mcq[mcq['CompensationAmount'].notnull()].shape


# In[71]:


# 자료 내에 존재하는 , or - 를 없앰
mcq['CompensationAmount'] = mcq[
    'CompensationAmount'].str.replace(',','')
mcq['CompensationAmount'] = mcq[
    'CompensationAmount'].str.replace('-','')

# 환율계산을 위한 정보 가져오기
rates = pd.read_csv('conversionRates.csv')
rates.head()


# In[72]:


# Unnamed: 0 지우기 위해 drop 사용
# inplace 초기 값 : False -> drop() 이 rates를 변경시키지 않음
#                   True  -> drop() 이 rates를 변경함. 즉, drop()실행 결과가 rates에 저장
rates.drop('Unnamed: 0',axis=1,inplace=True)
rates.head()


# In[73]:


# 언급한 컬럼만 뽑고 Nan 제거함(dropna())
salary = mcq[
    ['CompensationAmount','CompensationCurrency',
     'GenderSelect',
     'Country',
     'CurrentJobTitleSelect']].dropna()

# 좌측은 CompensationCurrency
# 우측은 originCountry 
# salary의 우측에 rates를 병합

salary = salary.merge(rates,left_on='CompensationCurrency',
                      right_on='originCountry', how='left')

print(salary.shape)
salary.head()


# In[74]:


# 숫자가 아닌 것을 숫자로 바꿔주는 to_numeric()메서드.
salary['Salary'] = pd.to_numeric(
    salary['CompensationAmount']) * salary['exchangeRate']
salary.head()


# In[75]:


print('Maximum Salary is USD $',
      salary['Salary'].dropna().astype(int).max())
print('Minimum Salary is USD $',
      salary['Salary'].dropna().astype(int).min())
print('Median Salary is USD $',
      salary['Salary'].dropna().astype(int).median())


# 가장 큰 수치는 여러 국가들의 GDP보다 크다고 한다. 가짜 응답이며, 평균급여는 USD $ 53,812 이다. 그래프를 좀 더 잘 표현하기 위해 50만불 이상의 데이터만 distplot으로 그려봤다.

# In[76]:


#시각화
plt.subplots(figsize=(15,8))
salary=salary[salary['Salary']<500000]
sns.distplot(salary['Salary'])
#중앙값을 점선으로 표시
plt.axvline(salary['Salary'].median(), linestyle='dashed')
plt.title('Salary Distribution',size=15)


# In[77]:


# 나라별 중앙값을 비교하여 어느 나라의 봉급이 더 많은지 확인
plt.subplots(figsize=(8,12))

sal_coun = salary.groupby(
    'Country')['Salary'].median().sort_values(
    ascending=False)[:30].to_frame()

sns.barplot('Salary', 
            sal_coun.index,
            data = sal_coun,
            palette='RdYlGn')

plt.axvline(salary['Salary'].median(), linestyle='dashed')
plt.title('Highest Salary Paying Countries')


# In[78]:


# 전 세계의 남성, 여성의 임금 차이 보여주는 boxplot
plt.subplots(figsize=(8,4))
sns.boxplot(y='GenderSelect',x='Salary', data=salary)


# In[79]:


# 한국의 경우 남여 임금 차이를 보여주는 boxplot
salary_korea = salary.loc[(salary['Country']=='South Korea')]
plt.subplots(figsize=(8,4))
sns.boxplot(y='GenderSelect',x='Salary',data=salary_korea)


# In[80]:


salary_korea.shape
# 총 26명이 응답한 것임


# In[81]:


salary_korea[salary_korea['GenderSelect'] == 'Female']
# 한국의 경우 여성 응답자가 3명, 너무 적은 숫자


# In[82]:


# 남성 봉급에 대한 요약
salary_korea_male = salary_korea[
    salary_korea['GenderSelect']== 'Male']
salary_korea_male['Salary'].describe()


# In[83]:


salary_korea_male


# ## Q9. 개인프로젝트나 학습용 데이터를 어디에서 얻나요? 

# In[84]:


mcq['PublicDatasetsSelect'] = mcq[
    'PublicDatasetsSelect'].astype('str').apply(
    lambda x: x.split(',')
    )


# In[85]:


q = mcq.apply(
    lambda x: pd.Series(x['PublicDatasetsSelect']),
    axis=1).stack().reset_index(level=1, drop=True)

q.name = 'courses'


# In[86]:


q = q[q != 'nan'].value_counts()


# In[87]:


pd.DataFrame(q)


# In[88]:


plt.title("Most Popular Dataset Platforms")
sns.barplot(y=q.index, x=q)


# Kaggle 및 Socrata는 개인 프로젝트나 학습에 사용하기 위한 데이터를 얻는데 인기있는 플랫폼이다. Google 검색 및 대학 / 비영리 단체 웹 사이트는 각각 2위와 3위에 있다. 그리고 직접 웹스크래핑 등을 통해 데이터를 수집한다고 한 응답이 4위다.

# In[89]:


# 주관식 응답을 읽어온다.
ff = pd.read_csv('freeformResponses.csv', 
                 encoding="ISO-8859-1", low_memory=False)
ff.shape


# In[90]:


# 설문내용과 누구에게 물어봤는지를 찾아봄
qc = question.loc[question[
    'Column'].str.contains('PersonalProjectsChallengeFreeForm')]
print(qc.shape)
qc.Question.values[0]


# ### Q. 개인프로젝트에서 공개된 데이터셋을 다루는 데 가장 어려운 점은 무엇일까?

# In[91]:


# 위에서 만든 ff에서 상위 15개만 가져옴
ppcff = ff[
    'PersonalProjectsChallengeFreeForm'].value_counts().head(15)
ppcff.name = '응답 수'
pd.DataFrame(ppcff)


# 대부분 데이터를 정제하는일이라고 응답하였고 그 다음이 데이터 크기다.

# In[92]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# %matplotlib inline 설정을 해주어야지만 노트북 안에 그래프가 디스플레이 된다.
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

def displayWordCloud(data = None, backgroundcolor = 'white', width=1200, height=600 ):
    wordcloud = WordCloud(stopwords = STOPWORDS, 
                          background_color = backgroundcolor, 
                         width = width, height = height).generate(data)
    plt.figure(figsize = (15 , 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show() 


# In[93]:


ppc = ff['PersonalProjectsChallengeFreeForm'].dropna()


# In[94]:


get_ipython().run_line_magic('time', "displayWordCloud(''.join(str(ppc)))")


# ## Q10. 데이터 사이언스 업무에서 가장 많은 시간을 필요로 하는 일은?

# In[95]:


time_features = [
    x for x in mcq.columns if x.find('Time') != -1][4:10]


# In[96]:


tdf = {}
for feature in time_features:
    tdf[feature[len('Time'):]] = mcq[feature].mean()

tdf = pd.Series(tdf)
print(tdf)
print()

plt.pie(tdf, labels=tdf.index, 
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title("Percentage of Time Spent on Each DS Job")
plt.show()


# ## Q11. 데이터사이언스 직업을 찾는데 가장 고려해야 할 요소는 무엇일까요?

# In[97]:


question = pd.read_csv('schema.csv')
question.shape


# In[98]:


# 설문내용과 누구에게 물어봤는지를 찾아봄
qc = question.loc[question[
    'Column'].str.contains('JobFactor')]
print(qc.shape)
qc.Question.values


# In[99]:


mcq = pd.read_csv('multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory=False)
job_factors = [
    x for x in mcq.columns if x.find('JobFactor') != -1]


# In[100]:


jfdf = {}
for feature in job_factors:
    a = mcq[feature].value_counts()
    a = a/a.sum()
    jfdf[feature[len('JobFactor'):]] = a

jfdf = pd.DataFrame(jfdf).transpose()

plt.figure(figsize=(6,10))
plt.xticks(rotation=60, ha='right')
sns.heatmap(jfdf.sort_values('Very Important', 
                             ascending=False), annot=True)


# In[101]:


jfdf.plot(kind='bar', figsize=(18,6), 
          title="Things to look for while considering Data Science Jobs")
plt.xticks(rotation=60, ha='right')
plt.show()


# 데이터 사이언티스트로 직업을 찾을 때 가장 고려할 요소는 배울 수 있는 곳인지, 사무실 근무환경, 프레임워크나 언어, 급여, 경영상태, 경력정도 순이다.

# ## Q12. 데이터 사이언티스트가 되기 위해 학위가 중요할까요?

# In[102]:


sns.countplot(y='UniversityImportance', data=mcq)


# In[103]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.figure_factory as fig_fact

top_uni = mcq['UniversityImportance'].value_counts().head(5)
top_uni_dist = []
for uni in top_uni.index:
    top_uni_dist.append(
        mcq[(mcq['Age'].notnull()) & \
            (mcq['UniversityImportance'] == uni)]['Age'])

group_labels = top_uni.index

fig = fig_fact.create_distplot(top_uni_dist,group_labels)
py.iplot(fig, filename='University Importance by Age')


# 마치 연령대 그래프를 찍어 본것과 같은 형태의 그래프다. 20~30대는 대학 학위가 매우 중요하다고 생각하며,
# 연령대가 높은 응답자들은 그다지 중요하지 않다고 응답했다.
# 300명 미만의 응답자만이 학위가 중요하지 않다고 생각한다.
# 
# 대부분의 응답자가 석사와 박사인 것을 고려해 봤을 때 이는 자연스러운 응답이다.

# ## Q13. 어디에서 부터 데이터사이언스를 시작해야 할까요?

# In[104]:


mcq[mcq['FirstTrainingSelect'].notnull()].shape


# In[105]:


sns.countplot(y='FirstTrainingSelect', data=mcq)


# 대부분의 응답자가 학사학위 이상으로 대학교육에 대한 중요성을 부여했지만,
# 가장 많은 응답자가 코세라, 유데미와 같은 온라인 코스를 통해 데이터 사이언스를 공부했고 그 다음으로 대학교육이 차지하고 있다.
# 
# 개인프로젝트를 해보는 것도 중요하다고 답한 응답자가 제법 된다.

# ## Q14. 데이터사이언티스트 이력서에서 가장 중요한 것은 무엇일까요?

# In[106]:


sns.countplot(y='ProveKnowledgeSelect', data=mcq)


# 머신러닝과 관련 된 직무경험이 가장 중요하고 다음으로 캐글 경진대회의 결과가 중요하다고 답했다.
# 그리고 온라인 강좌의 수료증이나 깃헙 포트폴리오 순으로 중요하다고 답했다.

# ## Q15. 머신러닝 알고리즘을 사용하기 위해 수학이 필요할까요?

# scikit과 같은 라이브러리는 세부 정보를 추상화하여 기본기술을 몰라도 ML 모델을 프로그래밍 할 수 있다. 그럼에도 그 안에 있는 수학을 아는 것이 중요할까?

# In[107]:


# 설문내용과 누구에게 물어봤는지를 찾아봄
qc = question.loc[question[
    'Column'].str.contains('AlgorithmUnderstandingLevel')]
qc


# In[108]:


mcq[mcq['AlgorithmUnderstandingLevel'].notnull()].shape


# In[109]:


sns.countplot(y='AlgorithmUnderstandingLevel', data=mcq)


# 현재 코딩업무를 하는 사람들에게 질문했으며, 기술과 관련 없는 사람에게 설명할 수 있는 정도라면 충분하다는 응답이 가장 많으며 좀 더디더라도 밑바닥부터 다시 코딩해 볼 수 있는 게 중요하다는 응답이 그 뒤를 잇는다.

# ## Q16. 어디에서 일을 찾아야 할까요?

# In[110]:


# 설문내용과 누구에게 물어봤는지를 찾아봄
question.loc[question[
    'Column'].str.contains(
    'JobSearchResource|EmployerSearchMethod')]


# In[111]:


plt.title("Best Places to look for a Data Science Job")
sns.countplot(y='JobSearchResource', data=mcq)


# 구직자들은 회사 웹사이트나 구직 사이트로부터 찾고 그 다음으로 특정 기술의 채용 게시판, 일반 채용 게시판, 친구나 가족, 이전 직장 동료나 리더를 통해 채용 정보를 얻는다.

# In[112]:


plt.title("Top Places to get Data Science Jobs")
sns.countplot(y='EmployerSearchMethod', data=mcq)


# 위에서 구직자는 주로 구직사이트로 부터 채용정보를 가장 많이 찾았으나, 채용자는 친구, 가족, 이전 직장 동료 등의 추천을 통해 가장 많이 사람을 구하며 다음으로 리쿠르터나 특정 회사에 소속 된 사람에게 직접 연락을 해서 구하는 비율이 높다.

# ### 그럼 한국 사람들은 어떨까?
# 

# In[113]:


korea = mcq.loc[(mcq['Country']=='South Korea')]
plt.title("Best Places to look for a Data Science Job")
sns.countplot(y='JobSearchResource', data=korea)


# In[114]:


plt.title("Top Places to get Data Science Jobs")
sns.countplot(y='EmployerSearchMethod', data=korea)


# # 결론 

# * 이 설문결과로 Python이 R보다 훨씬 많이 사용됨을 알 수 있었다.
# * 하지만 Python과 R을 모두 사용하는 사람도 많다.
# * 데이터 수집과 정제는 어려운 일이다.(공감)
# * 인기있는 학습플랫폼과 블로그, 유튜브 채널, 팟캐스트 등을 알게 되었다.
# * 내년에 인기있는 기술로는 딥러닝과 텐서플로우가 큰 차지를 할 것이다.
