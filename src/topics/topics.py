import gensim
import os
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import numpy as np
import pandas as pd
import cPickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)

data_folder = '../../data/'

topics_folder_fp = os.path.join(data_folder, '101 topics')

topics_files=['Business-Strategy.txt',
              'Humor.txt',
              'Software-Engineering.txt',
              'Interpersonal-Interaction.txt',
              'Tourism.txt',
              'Money.txt',
              'Product-Design-of-Physical-Goods.txt',
              'World-War-II.txt',
              'Quora.txt',
              'Education-Schools-and-Learning.txt',
              'Mathematics.txt',
              'Neuroscience-1.txt',
              'Hollywood.txt',
              'Restaurants.txt',
              'Tips-and-Hacks-for-Everyday-Life.txt',
              'Recipes.txt',
              'Investing.txt',
              'Writing.txt',
              'Design.txt',
              'Philosophy.txt',
              'Marriage.txt',
              'Career-Advice.txt',
              'International-Relations-3.txt',
              'The-College-and-University-Experience.txt',
              'Fine-Art.txt',
              'Government.txt',
              'The-Universe.txt',
              'Sports.txt',
              'Business-49.txt',
              'Fiction.txt',
              'Visiting-and-Travel-1.txt',
              'Technology-Trends.txt',
              'Politics.txt',
              'Self-Improvement.txt',
              'Health.txt',
              'Music.txt',
              'Mental-Health.txt',
              'Startup-Advice-and-Strategy.txt',
              'Entrepreneurship.txt',
              'Love.txt',
              'Social-Psychology.txt',
              'Television-Series.txt',
              'Dating-Advice.txt',
              'Cooking.txt',
              'Musicians.txt',
              'Web-Design.txt',
              'Digital-Photography.txt',
              'Venture-Capital.txt',
              'Programming-Languages.txt',
              'International-Travel.txt',
              'Computer-Science.txt',
              'Google-company-5.txt',
              'Healthy-Eating.txt',
              'Life-Lessons.txt',
              'Journalism.txt',
              'Exercise.txt',
              'Healthy-Living.txt',
              'Mobile-Phones.txt',
              'Vacations.txt',
              'Higher-Education.txt',
              'Startup-Founders-and-Entrepreneurs.txt',
              'Medicine-and-Healthcare.txt',
              'Rock-Music.txt',
              'Book-Recommendations.txt',
              'Facebook-product.txt',
              'Photography.txt',
              'Food.txt',
              'Nutrition.txt',
              'Life-Advice.txt',
              'Graduate-School-Education.txt',
              'Computer-Programming.txt',
              'Education.txt',
              'World-History.txt',
              'Lean-Startups.txt',
              'YouTube.txt',
              'Friendship.txt',
              'Reading-1.txt',
              'Marketing.txt',
              'Dating-and-Relationships-1.txt',
              'Literary-Fiction.txt',
              'Finance.txt',
              'Economics.txt',
              'Television.txt',
              'Learning.txt',
              'Clothing-and-Apparel.txt',
              'Startups.txt',
              'History.txt',
              'Science.txt',
              'Small-Businesses.txt',
              'Scientific-Research.txt',
              'User-Interfaces.txt',
              'Comedy.txt',
              'Movies.txt',
              'Physics.txt',
              'Literature.txt',
              'Algorithms.txt',
              'Religion.txt',
              'Social-Advice.txt',
              'Technology.txt',
              'Machine-Learning.txt',
              'History-of-the-United-States-of-America.txt']

label = 'label'
text = 'text'

def load_topics():
    texts=[]
    labels =[]
    for fname in topics_files:
        label = fname[:-4]
        l = open(os.path.join(topics_folder_fp, fname)).readlines()
        l = [x.strip() for x in l]
        texts+=l
        labels += len(l)*[label]

    return pd.DataFrame({'text':texts, 'label':labels})

