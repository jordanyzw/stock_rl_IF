#deep Q learning and policy gradient appilcation in stock market(IF index) domain
*first you need to install keras with tf backend and also install gym.
>
1.use gym and keras to implement the deep Q learning  and deep policy gradient  method.
>
2.the market_dqn.py is the implementation of DQN.
>
3.the market_pg.py is the implementatio of PG.
>
4.market_env.py is the implementation of the market env. it inherits the gym.env to simulate a market enviroment.
>
5.the result file is in dqn_result and pg_result. the number in each png means iteration times.
>
6.we get a good result and less time cost when we use PG algo.Atfer almost 100 epochs(takes almost half hour when use have a GPU)we can see that the result of trainging data set is good enough. And what mostly important is that the result of test data set is also stable and profitable. you can refer to the pg_result to see the final result.
I am sorry that I can not offer you the data because of some reasons.
