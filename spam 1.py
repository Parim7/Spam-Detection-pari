#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
data=pd.read_csv("C:/Users/uppad/Desktop/spam.csv", encoding="cp1252") # encoding tried in different ways in the end got it from google as cp1252
#its also working if u give any 4 didgits with cp


# In[5]:


data


# In[6]:


data.shape
data.info()


# In[7]:


#data.drop("Unnamed: 2",axis="columns",inplace=True) #by this code we are able to drop only 1 column


# In[8]:


#data=data.drop("Unnamed: 3", axis="columns")


# In[9]:


data.drop("Unnamed: 4", axis="columns")


# In[10]:


#data=data.drop(df.columns[[3,4]] axis=1, inplace= True)


# In[11]:


data.drop(data.iloc[:, 2:5], inplace=True, axis=1) #here I used iloc function to drop column from 3to 5 


# In[ ]:





# In[12]:


data


# In[13]:


data.sample(10)


# In[14]:


data.rename(columns={'v1': 'target', 'v2':'text'} , inplace= True) # renamed using dictionaries


# In[15]:


data


# In[16]:


from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()
data['target']= encoder.fit_transform(data['target']) 


# In[17]:


#data.head()


# In[18]:


#data=pd.get_dummies(data, drop_first= True) ## using dummies
##will also covert my text column into dummy so i used fit.transform to from Label encoder to convert specific"column"


# In[19]:


data


# In[20]:


data.isnull().sum()


# In[21]:


data.duplicated()
data.duplicated().sum()


# In[22]:


#remove
data=data.drop_duplicates(keep='first')


# In[23]:


data.info()


# In[24]:


data.shape


# EDA

# In[25]:


data.head()


# In[26]:


data['target'].value_counts()


# In[27]:


import matplotlib.pyplot as plt


# In[28]:


plt.pie(data['target'].value_counts(),labels=['ham','spam'],autopct='%0.2f')
plt.show


# In[29]:


import nltk


# In[30]:


pip install nltk


# In[31]:


nltk.download('punkt')


# In[32]:


data['num_characters']= data['text'].apply(len)


# In[33]:


data.head()


# In[34]:


data['text'].apply(lambda x:nltk.word_tokenize(x))
#data['text'].apply(lambda x:len(nltk.word_tokenize(x))


# In[35]:


data['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[36]:


data['num_words']=data['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[37]:


data.head()


# In[38]:


data['text'].apply(lambda x:nltk.sent_tokenize(x))


# In[39]:


data['num_sent']=data['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[40]:


data.sample(10)


# In[41]:


data[['num_characters','num_words' , 'num_sent']].describe()


# In[42]:


#we separated the target into spam and ham and then applied decribe to see thier properties
#ham
data[data['target']== 0][['num_characters','num_words' , 'num_sent']].describe()


# In[43]:


#spam
data[data['target']!= 0][['num_characters','num_words' , 'num_sent']].describe()
# data[data['target']==1][['num_characters','num_words' , 'num_sent']].describe()


# In[44]:


import seaborn as sns


# In[45]:


plt.figure(figsize=(12,9)) #size of the plot can be altered
sns.histplot(data[data['target']== 0]['num_characters'],color='1')
sns.histplot(data[data['target']== 1]['num_characters'],color="black")


# In[46]:


#here in the above plot we can see that ham msgs are made of many no of char compared to spam


# In[47]:


sns.histplot(data[data['target']== 0]['num_words'],color='blue')
sns.histplot(data[data['target']== 1]['num_words'],color="cyan")


# In[48]:


#here we can see that there are lot of outliers still ham are made of less words compared to spam msgs
#so we can plot corr coeff to know thier internal relation of columns


# In[49]:


sns.pairplot(data, hue="target") #hue represents the column u want to plot which is target here wrt other columns


# In[50]:


sns.heatmap(data.corr(), annot=True)
#here  in target column we are taking only spam values=1
#and our aim is here to compare corr with others columns wrt target column
#to build our model we need to take any 1 column so here we will choose num_char as 
#it has the largest correlation with target compared to other columns


# # Data Prepossesing
# 
# 

# 1. Lower case
# 2. Tokenize
# 3. Remove special characters
# 4. Removing stop words and punctiation
# 5. Stemming

# In[51]:


def transform_text (text):
    text =text.lower()
    text= nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    return y


# In[52]:


transform_text("Hi, i am parimala scored 568**%*")


# In[53]:


from nltk.corpus import stopwords
nltk.download("stopwords") ## i had to download stopwords as it was giving lookup error
stopwords.words("english")
##stopwords.words("Bengali") #I tried with bengal also


# In[54]:


import string
string.punctuation


# In[55]:


def transform_text (text):
    text =text.lower()           #converted text in lower case
    text= nltk.word_tokenize(text) # tokenized our text column
    y=[]
    for i in text:
        if i.isalnum():             #if words in text column are alphanumeric then append in y
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)                # here we are taking only those words which are not in stopwords  and punctuation list
    return y


# In[56]:


from nltk.stem.porter import PorterStemmer  #imported porterStemmer from nltk to apply stemming


# In[57]:


ps= PorterStemmer()    #named it as ps
ps.stem("dancing")      #applied stemming to dancing 


# In[58]:


def transform_text (text):
    text =text.lower()
    text= nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
            
    text=y[:]  #cleared the previous result   
    y.clear()
    
    for i in text:           # loop for stemming
        y.append(ps.stem(i))
    return " ".join(y)


# In[59]:


transform_text(" i loved doing this project got to learn many new things")


# In[60]:


data["transformed_text"] = data["text"].apply(transform_text)


# In[61]:


data.head()


# In[68]:


pip install wordcloud


# In[69]:


from wordcloud import WordCloud
wc = WordCloud(width=50,height=50,min_font_size=10,background_color='white')


# In[71]:


spam_wc = wc.generate(data[data['target']==1]['transformed_text'].str.cat(sep=""))


# In[72]:


plt.imshow(spam_wc)


# In[ ]:


- cf

