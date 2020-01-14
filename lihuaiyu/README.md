## 主要依赖的及版本
* tensorflow==2.0
* jieba
* scikit-learn

### 数据处理 
* 使用结巴分词
* 正则匹配非中文连续字符串为一个词，所有字符均保留，暂未剔除，后续进行分析
* 针对三任务分别进行剔除未标注0分类数据,去停用词
#### 利用 query 构建特征
* 搜索词条的数量
* 词条的平均长度、最大长度、最小长度
* 词条包含空格的比率
* 词条包含字母的比率

 

### 模型构建和评估
* 尝试使用了MultinomialNB 进行建模分析，分别对Age，Gender，Education 进行5折交叉检验，
模型检验准确度：
 <table>
        <tr>
            <th>k-fold</th>
            <th>1</th>
            <th>2</th>
            <th>3</th>
            <th>4</th>
            <th>5</th>
            <th>AVE</th>
        </tr>
        <tr>
            <th>Age</th>
            <th>0.527</th>
            <th>0.521</th>
            <th>0.529</th>
            <th>0.522</th>
            <th>0.515</th>
            <th>0.522</th>
        </tr>
        <tr>
             <th>Gender</th>
            <th>0.806</th>
            <th>0.804</th>
            <th>0.805</th>
            <th>0.803</th>
            <th>0.804</th>
            <th>0.804</th>
        </tr>
        <tr>
            <th>Education</th>
            <th>0.511</th>
            <th>0.522</th>
            <th>0.510</th>
            <th>0.516</th>
            <th>0.512</th>
            <th>0.514</th>
        </tr>
    </table>
    
关于使用MultinomialNB 模型进行训练消耗大量训练时间，和大量内存,强烈不建议 
    
* query特征模型构建正在进行中...

## 代码执行步骤
data_util.py 生成模型训练输入文件

run.py  模型训练





