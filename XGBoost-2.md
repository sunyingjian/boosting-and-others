## 算法原理

XGBoost是在Gradient Boosting框架下面对GBDT的优化，是一种GBDT的工业级实现。其主要原理是在GBDT的基础上，在损失函数加入正则化部分，并且每一轮迭代对误差无法做二阶泰勒展开，加快对损失函数的优化速度。下面基于决策树弱分类器，梳理XGBoost算法的主流程，部分参数可能现在会看得有些懵懂，后面的内容会慢慢提到，这里先给一个整体的流程框架。

* **输入：**

  * 训练数据集$T={(x_1,y_1),(x_2,y_2),...,(x_N,y_N)},x_i\in X\subseteq R^n,y_i \in Y={-1,+1}$；最大迭代次数$M$，损失函数 $L(Y,f(X))$；正则化系数$\lambda,\,\gamma$。
* **输出：**
* 强学习器 $f(x)$
* 初始化 $f_0(x)$
* 对$m=1,2,...,M$轮迭代有：

  * 1）计算第$i$个样本（$i=1,2,...,N$）在当前轮损失函数$L$基于$f_{m-1}(x_i)$的一阶导数$g_{mi}$，二阶导数$h_{mi}$，并求和：

    $$
    G_m=\sum_{i=1}^{N}g_{mi}

    $$

    $$
    H_m=\sum_{i=1}^{N}h_{mi}

    $$
  * 2）基于当前节点尝试分裂决策树，默认分数 $score=0$，对特征序号$k=1,2,...,K$

    * a）初始化$G_L=0,H_L=0$
    * b.1）将样本按特征 $k$ 从小到大排列，依次取出第 $i$个样本，依次计算当前样本放入左子树后，左右子树一阶和二阶导数和:

      $$
      G_L=G_L+g_{mi},\,G_R=G-G_L

      $$

      $$
      H_L=H_L+h_{mi},\,H_R=H-H_L

      $$
    * b.2）尝试更新最大的分数：

      $$
      score=max(score,\,\frac{1}{2}\frac{G_L^2}{H_L+\lambda}+\frac{1}{2}\frac{G_R^2}{H_R+\lambda}-\frac{1}{2}\frac{(G_L+G_R)^2}{H_L+H_R+\lambda})

      $$
* 3）基于最大$score$对应的划分特征和特征值分裂子树
* 4）如果最大$score$为0，则当前决策树建立完毕，计算所有叶子区域的$w_{mj}$，得到弱学习器$h_m(x)$，更新强学习器$f_m(x)$，进入下一轮弱学习器迭代。如果最大$score$不是0，则继续尝试分裂决策树。

## 损失函数

XGBoost的损失函数在GBDT$L(y,\,f_{m-1}(x)+h_m(x))$的基础上，加入了如下的正则化：

$$
\Omega(h_m)=\gamma J+\frac{\lambda}{2}\sum_{j=1}^{J}\omega_{mj}^2

$$

 这里的 $J$ 是叶子节点的个数，而 $\omega_{mj}$ 是第 $j$ 个叶子节点的最优值。这里的 $\omega_{mj}$ 和在GBDT里面使用的 $c_{mj}$ 其实是一个意思，只是[XGBoost论文](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1603.02754.pdf)里面使用 $\omega$ 符号表示叶子区域的值，这里为了和论文保持一致。

最终，XGBoost的损失函数可以表示为：

$$
L_m=\sum_{i=1}^{N}L(y_i, f_{m-1}(x_i)+h_m(x_i))+\gamma J+\frac{\lambda}{2}\sum_{j=1}^{J}\omega_{mj}^2

$$


如果要极小化上面这个损失函数，得到第$m$个决策树最优的所有$J$个叶子节点区域和每个叶子节点区域的最优解$\omega_{mj}$。XGBoost在损失函数优化方面做了一些优化，没有和GBDT一样去拟合泰勒展开式的一阶导数，而是期望直接基于损失函数的二阶泰勒展开式来求解。现在来看看这个损失函数的二阶泰勒展开式： 

$$
\begin{align} L_m &=\sum_{i=1}^{N}L(y_i, f_{m-1}(x_i)+h_m(x_i))+\gamma J+\frac{\lambda}{2}\sum_{j=1}^{J}\omega_{mj}^2 \\ & \approx \sum_{i=1}^{N}\left(L(y_i, f_{m-1}(x_i))+\frac{\partial L(y_i,\,f_{m-1}(x_i))}{\partial f_{m-1}(x_i)}h_m(x_i)+\frac{1}{2}\frac{\partial^2 L(y_i,\,f_{m-1}(x_i))}{\partial^2 f_{m-1}(x_i)}h_m^2(x_i)\right)+\gamma J+\frac{\lambda}{2}\sum_{j=1}^{J}\omega_{mj}^2 \end{align}

$$


为了方便起见，把第$i$个样本在第$m$个弱学习器的损失函数上面的一阶和二阶导数分别记为：

$$
g_{mi}=\frac{\partial L(y_i,\,f_{m-1}(x_i))}{\partial f_{m-1}(x_i)},\,h_{mi}=\frac{\partial^2 L(y_i,\,f_{m-1}(x_i))}{\partial^2 f_{m-1}(x_i)}

$$


则，损失函数可以简化为：

$$
L_m\approx \sum_{i=1}^{N}\left(L(y_i, f_{m-1}(x_i))+g_{mi}h_m(x_i)+h_{mi}h_m^2(x_i)\right)+\gamma J+\frac{\lambda}{2}\sum_{j=1}^{J}\omega_{mj}^2

$$


对于第$m$轮的迭代，损失函数里面$L(y_i,f_{m-1}(x_i))$为参数项，对最小化无影响，可以直接去掉。

定义

$$
I_j={i|q(x_i)=j}

$$

为第$j$个叶子节点的样本集合，经过$M$轮迭代完毕以后每个决策树（弱学习器）的第$j$个叶子节点的取值最终会是同一个值$\omega_{mj}$，因此，可以对损失函数继续化简：

$$
\begin{align} L_m &\approx \sum_{i=1}^{N}\left(g_{mi}h_m(x_i)+h_{mi}h_m^2(x_i)\right)+\gamma J+\frac{\lambda}{2}\sum_{j=1}^{J}\omega_{mj}^2 \\ & =\sum_{j=1}^{J}\left(\sum_{x_i \in R_{mj}}g_{mj}\omega_{mj}+\frac{1}{2}\sum_{x_i \in R_{mj}}h_{mj}\omega_{mj}^2\right)+\gamma J+\frac{\lambda}{2}\sum_{j=1}^{J}\omega_{mj}^2 \\ & = \sum_{j=1}^{J}[(\sum_{x_i \in R_{mj}}g_{mj})\omega_{mj}+\frac{1}{2}(\sum_{x_i \in R_{mj}}h_{mj}+\lambda)\omega_{mj}^2]+\gamma J \end{align}

$$

把每个叶子节点区域样本的一阶和二阶导数的和单独表示如下：

$$
G_{mj}=\sum_{x_i \in R_{mj}} g_{mj},\,H_{mj}=\sum_{x_i \in R_{mj}} h_{mj}

$$


那么最终的损失函数形式可以表示为：

$$
L_m=\sum_{j=1}^{J}[(G_{mj}\omega_{mj}+\frac{1}{2}(H_{mj}+\lambda)\omega_{mj}^2]+\gamma J

$$


现在有了最终的损失函数，回到前面讲到的问题，我们如何一次求解出决策树最优的所有$J$个叶子节点区域和每个叶子节点区域的最优解$\omega_{mj}$呢？

这个问题等价于下面两个子问题：

* 如果已经求出了第$m$个决策树的$J$个最优的叶子节点区域，如何求出每个叶子节点区域的最优解$\omega_{m }$？
* 对当前决策树做子树分裂决策时，应该如何选择哪个特征和特征值进行分裂，使最终的损失函数$ _m$最小？

第一个问题其实比较简单，如果树的结构$q(x)$已经确定的情况下，可以基于损失函数对$\omega_{m }$求偏导并令导数为0得出最佳的$\omega_{m }$，也就是每一个叶子节点的最优值表达式为：

$$
\omega_{m }=\frac{G_{mj}}{H_{mj}+\lambda}

$$


将求得的最优$\omega_{m }$代回到损失函数，可视为每轮迭代时该轮弱学习器的最优值表达式：

$$
L_m=-\frac{1}{2}\sum_{j=1}^{J}[\frac{G_{mj}}{H_{mj}+\lambda}]+\gamma J

$$


那么回到第二个问题，怎么样的树结构$q(x)$是最优的呢？

通常来讲，枚举所有可能的树的结构，然后计算该结构的分数是不太现实的。在实践当中，我们采用贪心的策略：

* 从深度为0的单一叶子节点开始，即所有的样本都在一个节点上
* 对于树的每一个叶子节点，尝试增加一个分裂点：

  * 令$I_L,\,I_R$分别表示分裂点加入之后的左右叶子节点的样本集合，则有
    * $$
      G_L=\sum_{i\in I_L}g_{mi}

      $$
    * $$
      G_R=\sum_{i\in I_R}g_{mi}

      $$
    * $$
      H_L=\sum_{i\in I_L}h_{mi}

      $$
    * $$
      H_R=\sum_{i\in I_R}h_{mi}

      $$
* 假如每次做左右子树分裂的时候，都可以最大程度地减少损失函数，则我们期望最大化下式：

  $$
  \Delta L=-\frac{1}{2}\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}+\gamma J-\left(-\frac{1}{2}\frac{G_L^2}{H_L+\lambda}-\frac{1}{2}\frac{G_R^2}{H_R+\lambda}+\gamma (J+1)\right)

  $$
* 对上式进行整理，我们期望最大化的是：

  $$
  L_{split}=\frac{1}{2}[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}]-\gamma

  $$

上式通常在实践当中用来评估分裂的候选点，也就是说，我们的决策树分裂标准不再使用CART树的均方误差，而是上式了。而具体如何分裂，将在下一节进行讨论。

## 分裂节点算法

## 精确搜索算法

* 对每一个节点，枚举所有的特征，对每一个特征，枚举所有的分裂点，目前大部分的GBDT算法实现（sklearn、R当中的GBDT）都是采用这种算法。
  * 对每个特征，通过特征的取值将实例进行排序
  * 寻求该特征的最优分裂点
  * 对所有的特征采用如下的算法操作

![](https://pic4.zhimg.com/80/v2-245267a2c87357969b87f21bd02c07a7_720w.jpg)

不难发现，对于连续型的特征，由于要枚举特征所有取值，计算量非常大，为了提高效率，通常将数据按照特征取值预排序。对于深度为$k$的树的时间复杂度：对特征所有取值的排序为$O(NlogN)$，$N$为样本点数目，若有$D$维特征，则$O(kDNlogN)$

## 近视搜索算法

精确搜索算法非常强大，但是当数据量大到内存无法存下时无法高效运行，并且在进行分布式计算时也存在一些问题，此时需要一个近视算法。主要思路如下： - 根据特征分布的百分位数，提出特征的一些**候选分裂点** - 将连续特征值映射到桶里（候选点对应的分裂），然后根据桶里样本的统计量，从这些候选中选择最佳分裂点 - 根据候选提出的时间，分为： - 全局近似：在构造树的初始阶段提出所有的候选分裂点，然后对各个层次采用相同的候选 - 提出候选的次数少，但每次的候选数目多（因为候选不更新） - 局部近似：在每次分裂都重新提出候选 - 对层次较深的树更适合，候选分裂点的数目不需要太多

对于$G,H$的更新算法步骤如下：

![](https://pic1.zhimg.com/80/v2-fe56ffd21e210dfded832df7400ab318_720w.jpg)

对于单机系统，XGBoost系统支持精确搜索算法，对于单机/分布式，全局近视/局部近视均支持。

## 正则化

总体上讲，XGBoost抵抗过拟合的方法与GBDT类似，主要也是从以下几方面考虑：

## 模型复杂度

正则化表达式为：

$$
\Omega(h_m)=\gamma J+\frac{\lambda}{2}\sum_{j=1}^{J}\omega_{mj}^2

$$

 这里的$J$是叶子节点的个数，而 $\omega_{mj}$ 是第$j$个叶子节点的最优值。

其中第一项叶子结点的数目$J$相当于$L_1$正则，第二项叶子节点的最优值相当于$L_2$正则。

## Shrinkage

其思想认为，每次走一小步逐渐逼近结果的效果，要比每次迈一大步很快逼近结果的方式更容易避免过拟合。即它不完全信任每一个棵残差树，它认为每棵树只学到了真理的一小部分，累加的时候只累加一小部分，通过多学几棵树弥补不足。用方程来看更清晰，即给每棵数的输出结果乘上一个步长$\alpha$（learning rate）

对于前面的弱学习器的迭代：

$$
f_m(x)=f_{m-1}(x)+T(x;\gamma_m)

$$

加上正则化项，则有

$$
f_m(x)=f_{m-1}(x)+\alpha\, T(x;\gamma_m)

$$

此处，$\alpha$的取值范围为(0,1]。对于同样的训练集学习效果，较小$\alpha$的意味着需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起决定算法的拟合效果。

## 特征采样和样本采样

XGBoost借鉴RF的思想，对于特征进行采样以达到降低过拟合的目的，根据用户反馈，特征采样比起样本采样效果更优。当然，XGBoost同时支持以上两种降低过拟合的采样方式。

## Early Stop

对于每一次分离后的增益，即前面的

$$
L_{split}=\frac{1}{2}[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}]-\gamma

$$


在`sklearn`接口当中，如果$L_{split}$出现负值，则提前停止；但是，被提前终止掉的分裂可能其后续的分裂会带来好处。在XBGoost原生接口当中是采用过后剪枝策略：将树分裂到最大深度，然后再基于上述增益计算剪枝。在具体实现当中还有`learning rate`等其它参数一起控制，给$L_{split}$出现负值的后续轮留机会。

## 缺失值处理

在实际的工业实践当中，数据出现缺失无法避免。

XGBoost没有假设缺失值一定进入左子树还是右子树，则是尝试通过枚举所有缺失值在当前节点是进入左子树，还是进入右子树更优来决定一个处理缺失值默认的方向，这样处理起来更加的灵活和合理。

也就是说，上面第1节的算法的步骤a），b.1）和b.2）会执行2次，第一次假设特征$k$所有有缺失值的样本都走左子树，第二次假设特征$k$所有缺失值的样本都走右子树。然后每次都是针对没有缺失值的特征k的样本走上述流程，而不是所有的的样本。

如果是所有的缺失值走右子树，使用上面第1节的a），b.1）和b.2）即可。如果是所有的样本走左子树，则上面第1节的a）步要变成：

$$
G_R=0,\,H_R=0

$$


b.1)步要更新为：

$$
G_R=G_R+g_mi,\,G_L=G-G_R

$$

 
$$
H_R=H_R+h_mi,\,H_L=H-H_R

$$


具体算法如下图所示：

![](https://pic4.zhimg.com/80/v2-2d39295c46c069ff9bfcd2b24e32de4f_720w.jpg)

## 优缺点

## 优点

* 支持线性分类器（相当于引入$L_1,\,L_2$正则惩罚项的LR和线性回归，损失函数公式=误差平方和+正则项，似LR）
* 损失函数用了二阶泰勒展开，引入一阶导和二阶导，提高模型拟和的速度
* 可以给缺失值自动划分方向
* 同RF，支持样本(行)随机抽取，也支持特征(列)随机抽取，降低运算，防过拟合
* 损失函数引入正则化项，包含全部叶子节点个数，每个叶子节点得分的$L_2$模的平方和（代表叶子节点权重的影响），控制模型（树）复杂度
* 每次迭代后为叶子分配结合学习速率，减低每棵树权重，减少每棵树影响，灵活调整后面的学习空间
* 支持并行，不是树并行，是把特征值先预排序，存起来，可以重复并行使用计算分裂点
* 分裂依据分开后与未分前的差值增益，不用每个节点排序算增益，减少计算量，可以并行计算
* 可以引入阈值限制树分裂，控制树的规模

## 缺点

* 往往会选择具有更高数量的不同值的预测器
* 当预测器具有很多类别时，容易过拟合
* 参数过多，调参困难
* 如果采用精确搜索算法对内存消耗过大，且不支持分布式

## sklearn中的参数解释

`sklearn`本身的文档当中并没有XGBoost的描述，[Github](https://link.zhihu.com/?target=https%3A//github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py)上面看到主要参数如下：

* max_depth ：树的最大深度。越大通常模型复杂，更容易过拟合，这里作为弱学习器一般设置为1-10
* learning_rate：学习率或收缩因子。学习率和迭代次数／弱分类器数目n_estimators相关。 缺省：0.1 （与直接调用xgboost的eta参数含义相同）
* n_estimators：弱分类器数目. 缺省:100
* verbosity：控制输出
* slient：参数值为1时，静默模式开启，不输出任何信息
* objective：待优化的目标函数，常用值有：

  * binary:logistic 二分类的逻辑回归，返回预测的概率
  * multi:softmax 使用softmax的多分类器，返回预测的类别(不是概率)
  * multi:softprob 和multi:softmax参数一样，但是返回的是每个数据属于各个类别的概率
  * 支持用户自定义目标函数
* booster：选择每次迭代的模型，有两种选择：
* gbtree：基于树的模型，为缺省值
* gbliner：线性模型
* dart：树模型
* nthread：用来进行多线程控制。 如果你希望使用CPU全部的核，那就用缺省值-1，算法会自动检测它
* n_jobs：xgboost运行时的线程数，后续代替`nthread`参数
* gamma：节点分裂所需的最小损失函数下降值，缺省0
* min_child_weight：叶子结点需要的最小样本权重（hessianhessian）和
* subsample：构造每棵树的所用样本比例（样本采样比例）
* colsample_bytree：构造树时每棵树的所用特征比例
* colsample_bylevel：构造树时每层的所用特征比例
* colsample_bynone：构造树时每次分裂的所用特征比例
* reg_alpha：$L_1$正则的惩罚系数
* reg_lambda：$L_2$正则的惩罚系数
* tree_method string [default= auto]：XGBoost中使用的树构造算法

  * XGBoost支持hist和大规模分布式训练，仅支持大规模外部内存版本。
  * 选择: auto, exact, approx, hist, gpu_hist
    * auto： 对于中小型数据集，将使用精确的贪婪（精确）
      * 对于非常大的数据集，将选择近似算法（近似）
      * 因为旧版本总是在单个机器中使用精确贪婪，所以当选择近似算法来通知该选择时，用户将得到消息
* exact: 精确的贪婪算法
* approx ： 使用分位数草图和梯度直方图的近似贪婪算法
* hist：快速直方图优化近似贪心算法
* gpu_hist：hist算法的GPU实现
* scale_pos_weight：正负样本的平衡，通常用于不均衡数据
* base_score：初始预测值
* random_state：随机种子
* missing：缺失值
* importance_type：特征重要程度计算方法

除了以上参数，XGBoost原生接口当中参数众多，主要有以下4大类：

* General parameters
* Booster parameters
* Learning task parameters
* Command line parameters(较少用到)

如果有遗漏，具体可以参阅[XGBoost Parameters](https://link.zhihu.com/?target=https%3A//xgboost.readthedocs.io/en/latest/parameter.html)

## 应用场景

XGBoost是GBDT框架的工业级优化实现，所有GBDT能够应用的场景，XGBoost都可以应用，所有回归问题（线性/非线性）、二分类问题（设定阈值，大于阈值为正例，反之为负例）。

按业务领域分，推荐系统、反欺诈、信用评级等。
