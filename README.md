# ARES_implementation
An implementation of the ARES -- the RNA tertiary structure prediction scoring function

The paper: https://www.science.org/doi/10.1126/science.abe5650

Since the author didn't provide the label file and the final checkpoint he used in the paper, so it's difficult to implement the code.

Struggling for weeks, I finally find out how to add labels in lmdb file. I put this code there for reference.

For covenience, comment is in Chinese.

Good luck!

这中间我还踩了很多坑，有时间的话我也把我踩过的坑发这上面填一下。吐槽一下这个ARES代码基本没法直接拿来用，还是需要改很多细节，一度让我很崩溃。给作者发邮件等了好久好久才得到回复说他们没提供······
大家加油！关于batchsize=16设置不成功的事情我找到了原因，但还没理解怎么解决。
