# Car-diagnostic-report-analysis-Chinese-Version-
This is my tensorflow2.0 implementation of the NLG  models applied to [Baidu competition](https://aistudio.baidu.com/aistudio/competition/detail/3): summarizing the car diagnostic reports for a mobile APP. The dataset contains a 110,000 car-related questions or problems provided by app users from real scenes, following by multi-round dialogues between technicians and app users. The main task is to filter the most import information based on conversations in order to build think tank for this mobile APP company. 

The below picture would be my ranking in this competition.

There are total 3 models applied for this competition: Traditional Seq2Seq + attention, PGN + Coverage, SIF + Transformer

My model-updating procedure:
-----------------------------
1) **Seq2Seq + attention**: 
The highest score (ROUGE_L) for this model would be near ***28.5***. However, there are still having significant amount of OOV and word-repetition problems. 
2) **PGN + Coverage**: 
Point-generater Network not only solves OOV problems, also reduces the vocab size and speeds up calculation convergence at the same time. The application of Coverage mechanism could visibly reduce word-repetition. The highest score would be near ***32***, increased by ***14%***.
3) **SIF + Transformer**: 
Since each piece of data contains the overall experience of a Q&A service (Brand,Model,Question,Dialogue), there are many useless information contributing to long sentences which is unfriendly to LSTM long-term memory, even Transformer. Therefore, build up sentence_embedding matrix based on SIF and create the sentence label due to their similarity with final result. Transformer is selected to train the data formed by sentence as unit. The f1_score for training set would be ***0.89***. Finally, after removing some redundant sentences, the score would be near ***35***, increased by ***9%***.
