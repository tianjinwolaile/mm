
#### 1.0 导入数据 #### 
path <-"D:\\wd\\document\\20240512"

setwd(path)

library(haven)
charls <- haven::read_dta("第一波次数据——有分数.dta")
table(charls$r1shlt)
# 1     2     3     4     5 
# 254  2162  3285 15492  4226 

temp <- as.data.frame(names(charls))


#### 2.0 模型分析-XGBoost #### 
### R与机器学习系列|11.梯度提升算法(Gradient Boosting)-XGBoost ###
## https://mp.weixin.qq.com/s/miJhAwrhuP7hViL0acya9w


### 除了传统的梯度提升和基于树的超参数，xgboost 还提供了额外的超参数，
# 可以帮助减少过拟合的可能性，从而提高预测准确性。



#### ---- xgboost需要一个特征矩阵和一个响应向量作为输入。
# 因此，为了提供特征矩阵的输入，我们需要将分类变量数值化（例如，独热编码，标签编码）。


  # Helper packages
  library(tidyverse)    # for general data wrangling needs
  
  # Modeling packages
  library(gbm)      # for original implementation of regular and stochastic GBMs
  library(h2o)      # for a java-based implementation of GBM variants
  library(xgboost)  # for fitting extreme gradient boosting
  library(rsample)# for data split
  library(shapviz)
  charls01 <- charls[,c("EI","ragender", "raeduc_c", "r1agey","r1ltact_c","r1socwk",
                         "h1atotfa","r1drinkn_c","r1smokef")]

  # No   Yes 
  # 21193  4226
  names(charls01)
  charls01 <- charls01
  
  data <- as.data.frame(charls01)
  # str(data)
  View(data)
  # data$ragender=as.factor(data$ragender)
  # data$raeduc_c=as.factor(data$raeduc_c)
  # data$r1agey=as.factor(data$r1agey)
  # data$r1ltact_c=as.factor(data$r1ltact_c)
  # data$r1socwk=as.factor(data$r1socwk)
  # data$h1atotfa=as.factor(data$h1atotfa)
  # data$r1drinkn_c=as.factor(data$r1drinkn_c)
  # data$r1smokef=as.factor(data$r1smokef)
  data$EI <- ifelse(charls01$EI <= 1.2, "No", "Yes")
  data$EI <- factor(data$EI, levels = c("No", "Yes"))

  # Stratified sampling with the rsample package
  set.seed(123)
  split <- initial_split(data, prop = 0.7, 
                         strata = "EI")
  data_train  <- training(split)
  data_test   <- testing(split)
  
  library(recipes)
  
  xgb_prep <- recipe(EI ~ ., data = data) %>%
    step_integer(all_nominal()) %>%  #数据预处理，将分类变量转变为数值变量
    prep(training = data_train, retain = TRUE) %>%
    juice()  # Extract transformed training set
  
  X <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "EI")])#特征变量矩阵
  Y <- xgb_prep$EI -1  ## 转换为0/1变量
 


#### ---- 网格搜索
set.seed(123)
data_xgb <- xgb.cv(
  data = X,
  label = Y,
  nrounds = 50,
  objective = "binary:logistic",
  early_stopping_rounds = 50, 
  nfold = 5,
  params = list(
    eta = 0.1,
    max_depth = 3,
    min_child_weight = 3,
    subsample = 0.8,
    colsample_bytree = 1.0),
  verbose = 0
)  

# minimum test CV RMSE
min(data_xgb$evaluation_log$test_logloss_mean)
## [1] 0.4319794


#### ---- 参数调整

  # hyperparameter grid
  hyper_grid <- expand.grid(
    eta = 0.01,
    max_depth = 3, 
    min_child_weight = 3,
    subsample = 0.5, 
    colsample_bytree = 0.5,
    gamma = c(0, 1, 10, 100, 1000),
    lambda = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
    alpha = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
    logloss = 0,          # a place to dump logloss results
    trees = 0          # a place to dump required number of trees
  )
  
  # grid search
  for(i in seq_len(nrow(hyper_grid))) {
    set.seed(123)
    m <- xgb.cv(
      data = X,
      label = Y,
      nrounds = 50,
      objective = "binary:logistic",
      early_stopping_rounds = 50, 
      nfold = 5,
      verbose = 0,
      params = list( 
        eta = hyper_grid$eta[i], 
        max_depth = hyper_grid$max_depth[i],
        min_child_weight = hyper_grid$min_child_weight[i],
        subsample = hyper_grid$subsample[i],
        colsample_bytree = hyper_grid$colsample_bytree[i],
        gamma = hyper_grid$gamma[i], 
        lambda = hyper_grid$lambda[i], 
        alpha = hyper_grid$alpha[i]
      ) 
    )
    hyper_grid$logloss[i] <- min(m$evaluation_log$test_logloss_mean)
    hyper_grid$trees[i] <- m$best_iteration
  }
  
  
  # results
  hyper_grid %>%
    filter(logloss > 0) %>%
    arrange(logloss) %>%
    glimpse()
  # Rows: 245
  # Columns: 10
  # $ eta              <dbl> 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,…
  # $ max_depth        <dbl> 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,…
  # $ min_child_weight <dbl> 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,…
  # $ subsample        <dbl> 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5…
  # $ colsample_bytree <dbl> 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5…
  # $ gamma            <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,…
  # $ lambda           <dbl> 1.00, 1.00, 0.00, 1.00, 0.10, 0.01, 0.10, 0.10, 0.00,…
  # $ alpha            <dbl> 0.10, 0.00, 0.10, 0.01, 0.10, 0.10, 0.01, 0.00, 0.00,…
  # $ logloss          <dbl> 0.4323468, 0.4323515, 0.4323526, 0.4323527, 0.4323555…
  # $ trees            <dbl> 498, 498, 498, 498, 498, 498, 498, 498, 498, 498, 498…
  
  # 选择logloss最小时的参数作为最佳参数用于最终模型建立
  {
    # optimal parameter list
    hyper_grid[86,] # logloss最小
    
    params <- list(
      eta = 0.01,
      max_depth = 3,
      min_child_weight = 3,
      subsample = 0.5,
      colsample_bytree = 0.5
    )
    # train final model
    xgb.fit.final <- xgboost(
      params = params,
      data = X,
      label = Y,
      nrounds = 500,
      objective = "binary:logistic",
      verbose = 0)
    }
  



#### ---- 变量重要性
# variable importance plot
library(ggsci)
mycol <- pal_npg("nrc", alpha = 0.5)(8) #提取6种颜色，透明度80%


vip::vip(xgb.fit.final,
         aesthetics = list(fill=mycol[1:8]))+
  theme_bw()


#### ---- ROC曲线

#测试集特征预处理
xgb_test <- recipe(EI ~ ., data = data) %>%
  step_integer(all_nominal()) %>%
  prep(training = data_test, retain = TRUE) %>%
  juice()

#测试集矩阵中预测
names(xgb_test)
data_test$predictions <- predict(xgb.fit.final,as.matrix(xgb_test[,-9]))


#绘制ROC曲线
library(pROC)
roc<- roc(EI~predictions,data=data_test)

library(ggplot2)
ggroc(roc,
      legacy.axes = T,
      size=1.5,
      color=mycol[1])+
  geom_abline(slope =1,intercept = 0,linewidth=1,color="grey")+
  theme_bw()+
  annotate(geom ="text",label=paste0("AUC in the test-set:",round(roc$auc,3)),x=0.6,y=0.2)



#### 3.0 模型分析-Shap #### 
### R与机器学习系列|15.可解释的机器学习算法（Interpretable Machine Learning） ###
## https://mp.weixin.qq.com/s/dj6xwwAP-k-y9jQLunViOQ


# Helper packages
library(tidyverse)    # for general data wrangling needs
# Modeling packages
library(gbm)      # for original implementation of regular and stochastic GBMs
library(h2o)      # for a java-based implementation of GBM variants
library(xgboost)  # for fitting extreme gradient boosting
library(rsample)# for data split
library(caret)# dummy funtion for categorical variables

path <- "D:\\wd\\document\\20240512\\五波次数据——含分数的\\第一波次数据——有分数.dta"
charls <- haven::read_dta(path)
charls01 <- charls[,c("EI","ragender", "raeduc_c", "r1agey","r1ltact_c","r1socwk",
                       "h1atotfa","r1drinkn_c","r1smokef")]

charls01$EI <- ifelse(charls01$EI <= 1.2, "No", "Yes")
charls01$EI <- factor(charls01$EI, levels = c("No", "Yes"))  ## 分值越大越不健康
# charls01$r1shlt_cat <- factor(charls01$r1shlt_cat,levels = c("0","1"))
table(charls01$EI)
# No   Yes 
# 21193  4226
names(charls01)

data <- as.data.frame(charls01)
# str(data)

# data$ragender=as.factor(data$ragender)
# data$raeduc_c=as.factor(data$raeduc_c)
# data$r1agey=as.factor(data$r1agey)
# data$r1ltact_c=as.factor(data$r1ltact_c)
# data$r1socwk=as.factor(data$r1socwk)
# data$h1atotfa=as.factor(data$h1atotfa)
# data$r1drinkn_c=as.factor(data$r1drinkn_c)
# data$r1smokef=as.factor(data$r1smokef)


# Stratified sampling with the rsample package
set.seed(123)
split <- initial_split(data, prop = 0.7, 
                       strata = "EI")
data_train  <- training(split)
data_test   <- testing(split)
data_train2=select(data_train, -EI)


### 独热编码 One-Hot Encoding
# 又称一位有效编码，其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，
# 并且在任意时候，其中只有一位有效。

dmytr = dummyVars(" ~ .", data =data_train2, fullRank=T)
data_train3 = predict(dmytr, newdata =data_train2)

X <-data_train3
Y<- as.numeric(data_train$EI) - 1  ## 转换为0/1变量
# table(Y)


# optimal parameter list
params <- list(
  eta = 0.01,
  max_depth = 3,
  min_child_weight = 3,
  subsample = 0.5,
  colsample_bytree = 0.5
)

# train final model
xgb.fit.final <- xgboost(
  params = params,
  data = X,
  label = Y,
  nrounds = 500,
  objective = "binary:logistic",
  verbose = 0
)
X=data.frame(X)
for(i in 1:(length(X))){
  X[,i]=as.numeric(X[,i])
}
shap=shapviz(xgb.fit.final,as.matrix(X))#对应模型和数据
sv_importance(shap)#对应shap结果
sv_importance(shap, kind = "no")
sv_importance(shap, kind = "beeswarm", show_numbers = TRUE)
sv_importance(shap, kind = "bar", show_numbers = TRUE)
sv_importance(shap, kind = "both", show_numbers = TRUE)

#### ---- 将特征重新由低到高进行标准化
feature_values <- X %>%
  as.data.frame() %>%
  mutate_all(scale) %>%
  gather(feature, feature_value) %>% 
  pull(feature_value)

#### ---- 计算特征的SHAP值以及SHAP重要性等参数
shap_df <- xgb.fit.final %>%
  predict(newdata = as.matrix(X), predcontrib = TRUE) %>%
  as.data.frame() %>%
  select(-BIAS) %>%
  gather(feature, shap_value) %>%
  mutate(feature_value = feature_values) %>%
  group_by(feature) %>%
  mutate(shap_importance = mean(abs(shap_value)))



#### ---- SHAP可视化
library(ggbeeswarm)
p1=ggplot(shap_df, aes(x = shap_value, y = reorder(feature, shap_importance))) +
  geom_quasirandom(groupOnX = FALSE, varwidth = TRUE, size =1, alpha = 0.8, aes(color = shap_value)) +
  scale_color_gradient(low = "#ffcd30", high = "#6600cd") +
  labs(x="SHAP value",y="")+
  theme_bw()+
  theme(axis.text = element_text(color = "black"),
        panel.border = element_rect(linewidth = 1))+
  geom_vline(xintercept = 0,linetype="dashed",color="grey",linewidth=1)




#### ---- 根据SHAP重要性值做一个SHAP重要性图
p2 <- shap_df %>% 
  select(feature, shap_importance) %>%
  filter(row_number() == 1) %>%
  ggplot(aes(x = reorder(feature, shap_importance), y = shap_importance,fill=feature)) +
  geom_col(alpha=0.6) +
  coord_flip() +
  xlab(NULL) +
  ylab("mean(|SHAP value|)")+
  # scale_fill_brewer(palette = "Paired")+
  theme_bw()+
  theme(legend.position = "",
        axis.text = element_text(color = "black"),
        panel.border = element_rect(linewidth = 1))




library(patchwork)
plot<-p1+p2&
  plot_layout(widths = c(2,1))
plot



#### ---- 基于Shapley值的依赖图
table(shap_df$feature)

shap_df %>% 
  filter(feature %in% c("r1agey")) %>%
  ggplot(aes(x = feature_value, y = shap_value)) +
  geom_point(aes(color = shap_value)) +
  scale_colour_viridis_c(name = "Feature value\n(standardized)", option = "C") +
  facet_wrap(~ feature, scales = "free") +
  scale_y_continuous('Shapley value', labels = scales::comma) +
  xlab('Normalized feature value')+
  theme_bw()



#### END #### 



