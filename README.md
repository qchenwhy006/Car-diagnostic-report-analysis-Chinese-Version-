# Car-diagnostic-report-analysis (Chinese-Version)
This is my tensorflow2.0 implementation of the NLG-related models applied to [Baidu competition](https://aistudio.baidu.com/aistudio/competition/detail/3): summarizing the car diagnostic reports for a mobile APP. The dataset contains a 110,000 questions or problems provided by app users from real scenes, following by multi-round dialogues between technicians and app users. The main task is to filter the most importance information based on conversations in order to build think tank for this mobile APP company. 

The below picture would be my rank (**<font size=5>8th, Second Prize</font>**) in this competition.
![Image](https://github.com/qchenwhy006/Car-diagnostic-report-analysis-Chinese-Version-/blob/master/Competition_ranking.png)

There are total 3 models applied for this competition: Traditional Seq2Seq + attention, PGN + Coverage, SIF + Transformer

My model-updating procedure:
-----------------------------
1) **Seq2Seq + attention**: 

The highest score (ROUGE-L) for this model would be near ***28***. However, there are still significant amount of OOV and word-repetition problems. 

2) **PGN + Coverage**: 

Point-generator Network not only could solve OOV problems, but also reduces the vocab size then speeds up calculation convergence. The application of Coverage strategy could visibly reduce word-repetition. The highest score would be near ***32***, increased by ***14%***.

3) **SIF + Transformer**: 

Since each piece of data contains the overall experience of a Q&A service (Brand, Model, Question, Dialogue), there are many useless information contributing to long sentences which is unfriendly to LSTM long-term memory, even Transformer. Therefore, built up sentence_embedding matrix based on SIF and created the sentence label due to their similarity with final result. Transformer was selected to train the data formed by sentence as unit. The f1_score for training set would be ***0.89***. Finally, after removing some redundant sentences, the score would be near ***35***, increased by ***25%***.


TODO:
-------------------------------
1) Since data contains two different types of question, one is question-form (e.g.: How mush does it cost on replacing a new tire? ) while the other is description of car situation. The former could be referenced by Reading Comprehension related models like Bi_DAF, DCN+ or S_Net. Span_extraction based task could accurately focus on the limit length of text, which is more precise than extractive-texting model. 
