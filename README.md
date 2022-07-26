# cookbook1401-6_9
Persian cookbook

# django related folders
## AIRProject 
its `__init__.py` run preprocesses and it is root of django project which indicate url patterns.

## FoodView 
convert food to a html file. all urls like `/FoodView/47` ended up being handled by this part.

## NextWords 
this part help to predict next words in the search bar.
first prediction address typos, the rest assume typed word are correct ,and it tries to complete it.

## searchEng 
return html of first page

## ShowResults 
by searching a text, this part, sends your text to the corresponding function and shows the result in a list format. 

## templates 
hold all html files

# searches folder (core part)

## data
hold all data files, like pretrained Kmean or dictionaries for boolean and tf-idf search.

## bert_bone
loads ParsBert model

## boolean, fasttext_search, transformer, tfidf
searches from HW3

## clustering
we use transformer as word2vec model this time and improve our results. 
by searching a text with this approach, system will initially search by transformer and concat the highest score food with 100 random food from its cluster. 

## classification (Ali)
we use transformer as word2vec model and train a decision tree on top, to classify foods in 5 different category. 
categories are: 
<ul>
<li> پیش غذا</li>
<li> کیک، شیرینی و نوشیدنی</li>
<li> غذای اصلی</li>
<li> سلامت غذایی</li>
<li> ساندویچ، پیتزا و ماکارونی،پاستا</li>
</ul>

We use a two layer fully connected as a classifier. The vectors from transformer are used as the input and the last layer specifies a distribution over classes.

## elastic_search (ehsan)
در ابتدا پیش‌پردازش را برا روی داده‌ها انجام می‌دهیم و سپس آن‌ها را در ایندکسی در دیتابیس الستیک ذخیره می‌کنیم.
برای سرچ نیز وزن بیشتری به نام غذا می‌دهیم و وزن کمتری را برای تگ‌ها و مواد اولیه در‌نظر میگیریم.
در اصل از روش function_score در الستیک استفاده کردیم. 

## next words prediction
by getting a text it tries to find a query for users.
### typo (Mahdi)

A language model is created and used for probability of words.

For each input query, all corpus words having edit distance at most 2 are generated.
Then the most probable one (according to language model and edit probability) is chosen as the result.

This spell corrected word is suggested and used in all searches.

### word prediction
we use ParsBert. it could predict masks words. we concat several [mask] sign to the text and then concat `"را میتوان خورد"`. then use bert to predict missing words which usually ended up being food. 

## preprocess 
it will preprocess food and normalizing them.




