title: LightGBM算法梳理 date: 2019-08-12 23:56:55 categories: 机器学习 tags: - 集成学习 - LightGBM - Histogram - 分布式 - sklearn description: DataWhale暑期学习小组-高级算法梳理第八期Task4。

---

## LightGBM

同XGBoost类似，LightGBM依然是在GBDT算法框架下的一种改进实现，是一种基于决策树算法的快速、分布式、高性能的GBDT框架，主要说解决的痛点是面对高维度大数据时提高GBDT框架算法的效率和可扩展性。

“Light”主要体现在三个方面，即更少的样本、更少的特征、更少的内存，分别通过单边梯度采样（Gradient-based One-Side Sampling）、互斥特征合并（Exclusive Feature Bundling）、直方图算法（Histogram）三项技术实现。另外，在工程上面，LightGBM还在并行计算方面做了诸多的优化，支持特征并行和数据并行，并针对各自的并行方式做了优化，减少通信量。

## LightGBM的起源

LightGBM起源于微软亚洲研究院在NIPS发表的系列论文：

* [Qi Meng, Guolin Ke, Taifeng Wang, Wei Chen, Qiwei Ye, Zhi-Ming Ma, Tie-Yan Liu. “A Communication-Efficient Parallel Algorithm for Decision Tree.” Advances in Neural Information Processing Systems 29 (NIPS 2016), pp. 1279-1287](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/6381-a-communication-efficient-parallel-algorithm-for-decision-tree.pdf)
* [Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. “LightGBM: A Highly Efficient Gradient Boosting Decision Tree.” Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3149-3157](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)

并于2016年10月17日在[LightGBM](https://link.zhihu.com/?target=https%3A//github.com/microsoft/LightGBM)上面开源，三天内在GitHub上面被star了1000次，fork了200次。知乎上现有近2000人关注“如何看待微软开源的LightGBM？”问题。

随后不断迭代，慢慢地开始支持Early Stopping、叶子索引预测、最大深度设置、特征重要性评估、多分类、类别特征支持、正则化（L1，L2及分裂最小增益）……

具体可以参阅以下链接：

[LightGBM大事记](https://link.zhihu.com/?target=https%3A//github.com/Microsoft/LightGBM/blob/master/docs/Key-Events.md)

## Histogram VS Pre-sorted

## Pre-sorted

在XGBoost当中的精确搜索算法（Exact Greedy Algorithm）在寻找分裂点时就是采用Pre-sorted的思想。具体过程不再赘述，可以参阅[XGBoost算法梳理](https://link.zhihu.com/?target=http%3A//datacruiser.io/2019/08/10/DataWhale-Workout-No-8-XGboost-Summary/)

预排序还是有一定优点的，如果不用预排序的话，在分裂节点的时候，选中某一个特征后，需要对A按特征值大小进行排序，然后计算每个阈值的增益，这个过程需要花费很多时间。

预排序算法在计算最优分裂时，各个特征的增益可以并行计算，并且能精确地找到分割点。但是预排序后需要保存特征值及排序后的索引，因此需要消耗两倍于训练数据的内存，时间消耗大。另外预排序后，特征对梯度的访问是一种随机访问，并且不同的特征访问的顺序不一样，无法对cache进行优化，时间消耗也大。最后，在每一层，需要随机访问一个行索引到叶子索引的数组，并且不同特征访问的顺序也不一样。

## Historgram

首先需要指出的是，XGBoost在寻找树的分裂节点的也是支持直方图算法的，就是论文中提到的近视搜索算法（Approximate Algorithm）。只是，无论特征值是否为0，直方图算法都需要对特征的分箱值进行索引，因此对于大部分实际应用场景当中的稀疏数据优化不足。

回过头来，为了能够发挥直方图算法的优化威力，LightGBM提出了另外两个新技术：单边梯度采样（Gradient-based One-Side Sampling）和互斥特征合并（Exclusive Feature Bundling），在减少维度和下采样上面做了优化以后才能够将直方图算法发挥得淋漓尽致。下面依次介绍直方图算法、GOSS和EFB。

### 直方图算法

直方图算法的基本思想是先把连续的浮点特征值离散化成$k$个整数，同时构造一个宽度为$k$的直方图。在遍历数据的时候，根据离散化后的值作为索引在直方图中累积统计量，当遍历一次数据后，直方图累积了需要的统计量，然后根据直方图的离散值，遍历寻找最优的分割点。具体算法描述如下：

![](https://pic4.zhimg.com/80/v2-5f47156c3c7bed03f6645e2261775c7b_720w.jpg)

![](https://pic4.zhimg.com/80/v2-deb1a0b3ff7c4f13777e6866845ce1f3_720w.jpg)

直方图算法有如下优点：

* 内存消耗降低。预排序算法需要的内存约是训练数据的两倍（2x样本数x维度x4Bytes），它需要用32位浮点来保存特征值，并且对每一列特征，都需要一个额外的排好序的索引，这也需要32位的存储空间。对于 直方图算法，则只需要(1x样本数x维度x1Bytes)的内存消耗，仅为预排序算法的1/8。因为直方图算法仅需要存储特征的 bin 值(离散化后的数值)，不需要原始的特征值，也不用排序，而bin值用8位整型存储就足够了。
* 算法时间复杂度大大降低。决策树算法在节点分裂时有两个主要操作组成，一个是“寻找分割点”，另一个是“数据分割”。从算法时间复杂度来看，在“寻找分割点”时，预排序算法对于深度为$k$的树的时间复杂度：对特征所有取值的排序为$O(NlogN)$，$N$为样本点数目，若有$D$维特征，则$O(kDNlogN)$，而直方图算法需要$O(kD \times bin)$(bin是histogram 的横轴的数量，一般远小于样本数量$N$)。
* 再举个例子说明上述两点的优化，假设数据集$A$的某个特征的特征值有（二叉树存储）：${1.2,1.3,2.2,2.3,3.1,3.3}$，预排序算法要全部遍历一遍，需要切分大约5次。进行离散化后，只需要切分2次 ${{1},{2,3}}$ 和 ${{1,2},{3}}$，除了切分次数减少，内存消耗也大大降低。
* 直方图算法还可以进一步加速。一个容易观察到的现象：一个叶子节点的直方图可以直接由父节点的直方图和兄弟节点的直方图做差得到。通常构造直方图，需要遍历该叶子上的所有数据，但直方图做差仅需遍历直方图的$k$个bin。利用这个方法，LightGBM可以在构造一个叶子的直方图后，可以用非常微小的代价得到它兄弟叶子的直方图，在速度上可以提升一倍。

![](https://pic1.zhimg.com/80/v2-86919e4fc187a11fe3fdb72780709c98_720w.jpg)

当然，直方图算法并不是完美的。由于特征被离散化后，找到的并不是很精确的分割点，所以会对结果产生影响。但在不同的数据集上的结果表明，离散化的分割点对最终的精度影响并不是很大，甚至有时候会更好一点。原因是决策树本来就是弱模型，分割点是不是精确并不是太重要；较粗的分割点也有正则化的效果，可以有效地防止过拟合；即使单棵树的训练误差比精确分割的算法稍大，但在梯度提升（GradientBoosting）的框架下没有太大的影响。

### 单边梯度采样（Gradient-based One-Side Sampling）

单边梯度采样是一种在减少数据量和保证精度上平衡的算法。在GBDT中， 我们对损失函数的负梯度进行拟合，样本误差越大，梯度的绝对值越大，这说明模型对该样本的学习还不足够，相反如果越小则表示模型对该样本的学习已经很充分。因此，我们可以这样推论：梯度的绝对值越大，样本重要性越高。单边梯度采样正是以样本的梯度作为样本的权重进行采样。

单边梯度采样保留所有的梯度较大的样本，在梯度小的实例上使用随机采样。为了抵消对数据分布的影响，计算信息增益的时候，单边梯度采样对小梯度的数据引入常量乘数。单边梯度采样首先根据数据的梯度绝对值排序，选取$top\,\,a$个样本。然后在剩余的数据中随机采样$b$个样本。接着计算信息增益时为采样出的小梯度数据乘以$\frac{1-a}{b}$，这样算法就会更关注训练不足的样本例，而不会过多改变原数据集的分布。正是单边梯度采样减少了数据量，让直方图算法发挥了更大的作用。单边梯度采样具体算法见下图：

![](https://pic2.zhimg.com/80/v2-d891d643d36990bafd0a3ad0160910fd_720w.jpg)

### 互斥特征合并（Exclusive Feature Bundling）

高维的数据通常是稀疏的，这种特征空间的稀疏性给我们提供了一种设计一种接近无损地降维的可能性。特别的，在稀疏特征空间中，许多特征是互斥的，换句话说，大部分特征不会同时取非0值，例如One-hot之后的类别特征他们从不同时为非0值。我们可以合并互斥的特征为单一特征（我们将这个过程称为*Exclusive Feature Bundle* ）.通过仔细设计特征扫描算法，我们从合并特征中构建出与单个特征相同的特征直方图。通过这种方式，直方图构建算法的时间复杂度从$O(ND)$，降到$O(N×bundle)$，其中$N$为样本点数目，$D$为特征维度。通常，bundle << $D$，我们能够在不损失精度的情况下极大地加速GBDT的训练。

那么接下来有两个问题需要处理：

* 需要合并哪些特征
* 如何合并这些特征

### Greedy Bundling

找出最优的 bundle 组合数是一个 NP 问题，LightGBM通过将原问题转化为”图着色”问题进行贪心近似解决。

首先创建一个图$G(V, E)$，其中$V$就是特征，为图中的节点，$E$为$G$中的边，将不是相互独立的特征用一条边连接起来，边的权重就是两个相连接的特征的总冲突值，这样需要绑定的特征就是在图着色问题中要涂上同一种颜色的那些点（特征）。具体算法过程如下：

![](https://pic2.zhimg.com/80/v2-ff283da785373ccb98b4f891b2355005_720w.jpg)

### Merge Exclusive Features

该过程主要是关键在于原始特征值可以从bundle中区分出来，即绑定几个特征在同一个bundle里需要保证绑定前的原始特征的值在bundle中能够被识别，考虑到直方图算法将连续的值保存为离散的bin，我们可以使得不同特征的值分到bundle中的不同bin中，这可以通过在特征值中加一个偏置常量来解决，比如，我们在bundle中绑定了两个特征A和B，A特征的原始取值为区间[0,10)，B特征的原始取值为区间[0,20），我们可以在B特征的取值上加一个偏置常量10，将其取值范围变为[10,30），这样就可以放心的融合特征A和B了，因为在树模型中对于每一个特征都会计算分裂节点的，也就是通过将他们的取值范围限定在不同的bin中，在分裂时可以将不同特征很好的分裂到树的不同分支中去。具体算法如下：

![](https://pic2.zhimg.com/80/v2-218502a2ad06b2d61347b335fe1458a9_720w.jpg)

## Leaf-wise VS Level-wise

## Level-wise

大多数GBDT框架使用的按层生长 (level-wise) 的决策树生长策略，Level-wise遍历一次数据可以同时分裂同一层的叶子，容易进行多线程优化，也好控制模型复杂度，不容易过拟合。但实际上Level-wise是一种低效的算法，因为它不加区分的对待同一层的叶子，带来了很多没必要的开销，因为实际上很多叶子的分裂增益较低，没必要进行搜索和分裂。

![](https://pic4.zhimg.com/80/v2-ab9552751492a866f37dfe2944d86e2f_720w.jpg)

## Leaf-wise

LightGBM在直方图算法之上，对于树的生长策略做了进一步优化，抛弃了Level-wise策略，使用了带有深度限制的按叶子生长 (leaf-wise)算法。Leaf-wise则是一种更为高效的策略，每次从当前所有叶子中，找到分裂增益最大的一个叶子，然后分裂，如此循环。因此同Level-wise相比，在分裂次数相同的情况下，Leaf-wise可以降低更多的误差，得到更好的精度。Leaf-wise的缺点是可能会长出比较深的决策树，产生过拟合。因此LightGBM在Leaf-wise之上增加了一个最大深度的限制，在保证高效率的同时防止过拟合。

![](https://pic2.zhimg.com/80/v2-e5520fb431e867af60c16de62eedf85d_720w.jpg)

## 特征并行和数据并行

本小节主要根据[LightGBM的官方文档](https://link.zhihu.com/?target=https%3A//lightgbm.readthedocs.io/en/latest/Features.html%23optimization-in-network-communication)中提到的并行计算优化进行讲解。 在本小节中，工作的节点称为**worker**

LightGBM具有支持高效并行的优点。LightGBM原生支持并行学习，目前支持特征并行和数据并行的两种。

## 特征并行

特征并行主要是并行化决策树中寻找最优划分点(“Find Best Split”)的过程，因为这部分最为耗时。

传统特征并行做法如下：

* 垂直划分数据（对特征划分），不同的worker有不同的特征集
* 每个workers找到局部最佳的切分点{feature, threshold}
* workers使用点对点通信，找到全局最佳切分点
* 具有全局最佳切分点的worker进行节点分裂，然后广播切分后的结果（左右子树的instance indices）
* 其它worker根据收到的instance indices也进行划分

![](https://pic3.zhimg.com/80/v2-b0d10c5cd832402e4503e2c1220f7376_720w.jpg)

主要有以下缺点：

* 无法加速分裂的过程，该过程的时间复杂度为$O(N)$，当数据量大的时候效率不高
* 需要广播划分的结果（左右子树的instance indices），1条数据1bit的话，通信花费大约需要$O(N/8)$

LightGBM的特征并行每个worker保存所有的数据集，这样找到全局最佳切分点后各个worker都可以自行划分，就不用进行广播划分结果，减小了网络通信量。过程如下：

* 每个workers找到局部最佳的切分点{feature, threshold}
* workers使用点对点通信，找到全局最佳切分点
* 每个worker根据全局全局最佳切分点进行节点分裂

这样虽然不用进行广播划分结果，减小了网络通信量。但是也有缺点：

* 分裂过程的复杂度保持不变
* 每个worker保存所有数据，存储代价高

## 数据并行

数据并行的目标是并行化整个决策学习的过程，传统算法的做法如下：

* 水平切分数据，不同的worker拥有部分数据
* 每个worker根据本地数据构建局部直方图
* 合并所有的局部直方图得到全部直方图
* 根据全局直方图找到最优切分点并进行分裂

![](https://pic4.zhimg.com/80/v2-0b80a0229c2a45c62c98dd41e1cc63c7_720w.jpg)

在第3步当中有两种合并方式：

* 采用点对点方式(point-to-point communication algorithm)进行通讯，每个worker通讯量为$ ( ℎ × × )$
* 采用collective communication algorithm(如“[All Reduce](https://link.zhihu.com/?target=http%3A//pages.tacc.utexas.edu/~eijkhout/pcse/html/mpi-collective.html)”)进行通讯（相当于有一个中心节点，通讯后在返回结果），每个worker的通讯量为$O(2× × )$

不难发现，通信的代价也是很高的，这也是数据并行的缺点。

LightGBM的数据并行主要做了以下两点优化：

* 使用“Reduce Scatter”将不同worker的不同特征的直方图合并，然后workers在局部合并的直方图中找到局部最优划分，最后同步全局最优划分
* 通过直方图作差法得到兄弟节点的直方图，因此只需要通信一个节点的直方图，减半通信量

通过上述两点做法，通信开销降为$ (0.5 × × )$。

另外，LightGBM还采用 一种称为**PV-Tree** 的算法进行投票并行（Voting Parallel），其实本质上也是一种数据并行。

PV-Tree和普通的决策树差不多，只是在寻找最优切分点上有所不同。具体算法如下：

![](https://pic3.zhimg.com/80/v2-4812b3c08fa0dd01d3840c649aa38d4e_720w.jpg)

主要思路如下：

* 水平切分数据，不同的worker拥有部分数据
* Local voting: 每个worker构建直方图，找到$top-k$个最优的本地划分特征
* Global voting: 中心节点聚合得到最优的$top-2k$个全局划分特征（$top-2k$是看对各个worker选择特征的个数进行计数，取最多的$2k$个）
* Best Attribute Identification： 中心节点向worker收集这$top-2k$个特征的直方图，并进行合并，然后计算得到全局的最优划分
* 中心节点将全局最优划分广播给所有的worker，worker进行本地划分

![](https://pic2.zhimg.com/80/v2-9a2c6161fa1bdacd0f27d237ef06f2ed_720w.jpg)

可以看出，PV-tree将原本需要$O( × )$ 变为了$O(2k × )$，通信开销得到降低。此外，可以证明，当每个worker的数据足够多的时候，$top-2k$个中包含全局最佳切分点的概率非常高。

## 顺序访问梯度

Cache（高速缓存）作为内存局部区域的副本，用来存放当前活跃的程序和数据，它利用程序运行的局部性，把局部范围的数据从内存复制到Cache中，使CPU直接高速从Cache中读取程序和数据，从而解决CPU速度和内存速度不匹配的问题。高速缓冲存储器最重要的技术指标是它的命中率。CPU在Cache中找到有用的数据被称为命中，当Cache中没有CPU所需的数据时（这时称为未命中），CPU才访问内存。

预排序算法中有两个频繁的操作会导致cache-miss，也就是缓存消失（对速度的影响很大，特别是数据量很大的时候，顺序访问比随机访问的速度快4倍以上 ）。

* 对梯度的访问：在计算增益的时候需要利用梯度，假设梯度信息存在一个数组$g[i]$中，在对特征$A$进行增益时，需要根据特征$A$排序后的索引找到$g[i]$中对应的梯度信息。特征值$A_1$对应的样本行号可能是3，对应的梯度信息在$g[3]$，而特征值$A_2$对应的样本行号可能是9999，对应的梯度信息在$g[9999]$，即对于不同的特征，访问梯度的顺序是不一样的，并且是随机的
* 对于索引表的访问：预排序算法使用了行号和叶子节点号的索引表，防止数据切分的时候对所有的特征进行切分。同访问梯度一样，所有的特征都要通过访问这个索引表来索引

这两个操作都是随机的访问，会给系统性能带来非常大的下降。

LightGBM使用的直方图算法能很好的解决这类问题。首先。对梯度的访问，因为不用对特征进行排序，同时，所有的特征都用同样的方式来访问，所以只需要对梯度访问的顺序进行重新排序，所有的特征都能连续的访问梯度。并且直方图算法不需要把数据id到叶子节点号上（不需要这个索引表，没有这个缓存消失问题），大大提高cache的命中率，减少cache-miss出现的概率。

## 支持类别特征

实际上大多数机器学习工具都无法直接支持类别特征，一般需要把类别特征，转化one-hot特征，降低了空间和时间的效率。而类别特征的使用是在实践中很常用的。基于这个考虑，LightGBM优化了对类别特征的支持，可以直接输入类别特征，不需要额外的0/1展开。并在决策树算法上增加了类别特征的决策规则，直接原生支持类别特征，不需要转化，提高了近8倍的速度。

![](https://pic2.zhimg.com/80/v2-601009229abc3470afd0ed5e69d2b30d_720w.jpg)

## 应用场景

作为GBDT框架内的算法，GBDT、XGBoost能够应用的场景LightGBM也都适用，并且考虑到其对于大数据、高维特征的诸多优化，在数据量非常大、维度非常多的场景更具优势。近来，有着逐步替代XGBoost成为各种数据挖掘比赛baseline的趋势。

## sklearn参数

`sklearn`本身的文档当中并没有LightGBM的描述，[Github](https://link.zhihu.com/?target=https%3A//github.com/microsoft/LightGBM/blob/master/python-package/lightgbm/sklearn.py)上面看到主要参数如下：

* `boosting_type` : 提升类型，字符串，可选项 (default=`gbdt`)

  * `gbdt`, 传统梯度提升树
  * `dart`, 带Dropout的MART
  * `goss`, 单边梯度采样
  * `rf`, 随机森林
* `num_leaves` : 基学习器的最大叶子树，整型，可选项 (default=31)
* `max_depth` : 基学习器的最大树深度，小于等于0表示没限制，整型，可选项 (default=-1)
* `learning_rate` : 提升学习率，浮点型，可选项 (default=0.1)
* `n_estimators` : 提升次数，整型，可选项 (default=100)
* `subsample_for_bin` : 构造分箱的样本个数，整型，可选项 (default=200000)
* `objective` : 指定学习任务和相应的学习目标或者用户自定义的需要优化的目标损失函数，字符串， 可调用的或者None, 可选项 (default=None)，若不为None，则有:
* `regression` for LGBMRegressor -`binary` or `multiclass` for LGBMClassifier
* `lambdarank` for LGBMRanker
* `class_weight` : 该参数仅在多分类的时候会用到，多分类的时候各个分类的权重，对于二分类任务，你可以使用`is_unbalance` 或 `scale_pos_weight`，字典数据, `balanced` or None, 可选项 (default=None)
* `min_split_gain` : 在叶子节点上面做进一步分裂的最小损失减少值，浮点型，可选项 (default=0.)
* `min_child_weight` : 在树的一个孩子或者叶子所需的最小样本权重和，浮点型，可选项 (default=1e-3)
* `min_child_samples` : 在树的一个孩子或者叶子所需的最小样本，整型，可选项 (default=20)
* `subsample` : 训练样本的子采样比例，浮点型，可选项 (default=1.)
* `subsample_freq` : 子采样频率，小于等于0意味着不可用，整型，可选项 (default=0)
* `colsample_bytree` : 构建单棵树时列采样比例，浮点型，可选项 (default=1.)
* `reg_alpha` : $L_1$正则项，浮点型，可选项 (default=0.)
* `reg_lambda` :$L_2$正则项，浮点型，可选项 (default=0.)
* `random_state` : 随机数种子，整型或者None, 可选项 (default=None)
* `n_jobs` : 线程数，整型，可选项 (default=-1)
* `silent` : 运行时是否打印消息，布尔型，可选项 (default=True)
* `importance_type` : 填入到`feature_importances_`的特征重要性衡量类型，如果是`split`，则以特征被用来分裂的次数，如果是`gain`，则以特征每次用于分裂的累积增益，字符串，可选项 (default=`split`)

除了以上参数，LightGBM原生接口当中参数众多，主要有以下八大类：

* 核心参数
* 学习控制参数
* IO参数
* 目标参数
* 度量参数
* 网络参数
* GPU参数
* 模型参数

如果有遗漏，具体可以参阅[LightGBM Parameters](https://link.zhihu.com/?target=https%3A//lightgbm.readthedocs.io/en/latest/Parameters.html)

## CatBoost（了解）

CatBoost也是Boosting族的算法，由俄罗斯科技公司Yandex于2017年提出，主要在两方面做了优化，一个是对于类别变量的处理，另外一个是对于预测偏移（prediction shift）的处理。

其中对于类别变量在传统的Greedy TBS方法的基础上添加先验分布项，这样可以减少减少噪声和低频率数据对于数据分布的影响： 

$$
\hat{x}*k^i=\frac{\sum* {j=1}^n I_{{x_j^i=x_k^i}}*y_j+a\,P}{\sum_{j=1}^n I_{{x_j^i=x_k^i}}+a}

$$


其中 $P$ 是添加的先验项，$a$ 通常是大于 0 的权重系数。

对于第二个问题，CatBoost采用了排序提升（Ordered Boosting）的方式，首先对所有的数据进行随机排列，然后在计算第 $i$ 步残差时候的模型只利用了随机排列中前 $i-1$ 个样本。具体算法描述请参阅论文[CatBoost: unbiased boosting with categorical features](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features.pdf)

时间有限，下次有机会再详细消化下CatBoost的论文。

总之，CatBoost大大简化了前期数据处理过程，特别是类别特征的数值化，调参也相对容易。近来在数据竞赛领域已经大规模采用。
