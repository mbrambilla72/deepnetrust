# deepnetrust

An experimental port to rust from c# dnn code based on articles : \

https://visualstudiomagazine.com/articles/2015/04/01/back-propagation-using-c.aspx \

https://docs.microsoft.com/en-us/archive/msdn-magazine/2017/august/test-run-deep-neural-network-io-using-csharp \

versions \
\
0.1.0 \
initial port \
\
0.1.1 \
rand -> oorandom \
add benchmark tanh \
more costants for array use to see speedup \
simplify & clean \
change for to iterator (for_each) \
apply clippy suggests \
new faster tanh \
speedup from 26s -> 5secs : 5X more fast \
