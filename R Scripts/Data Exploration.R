library("rpart")
library("rpart.plot")
library("caret")
library("randomForest")
library("varImp")
library("xgboost")

matches_df <- read.csv("C:/Users/Lee Seung Soo/OneDrive/Desktop/Football Project/matches_rolling.csv")

colnames(matches_df)

str(matches_df)

#Changing the venue code, team code, opponent code and target to categorical
matches_df$venue_code <- as.factor(matches_df$venue_code)
matches_df$team_code <- as.factor(matches_df$team_code)
matches_df$opp_code <- as.factor(matches_df$opp_code)
matches_df$target <- as.factor(matches_df$target)

#Fitting the random forest model with all the possible variables
regressor <- randomForest(target ~ venue_code + opp_code + team_code + gf_rolling_3 + ga_rolling_3 +
                          sh_rolling_3 + sot_rolling_3 + dist_rolling_3 + form_rolling_3 + xg_rolling_3 + poss_rolling_3 + 
                          sota_rolling_3 + save._rolling_3 + cs_rolling_3 + psxg_rolling_3 + cmp_rolling_3 + cmp._rolling_3 +
                          prgdist_rolling_3 + ast_rolling_3 + ppa_rolling_3 + prog_rolling_3 + sca_rolling_3 + gca_rolling_3 +
                          tklw_rolling_3 + int_rolling_3 + tkl.int_rolling_3 + err_rolling_3 + succ_rolling_3 + succ._rolling_3 +
                          crdy_rolling_3 + fls_rolling_3 + won._rolling_3,
                          importance = TRUE, data = matches_df)
varImp(regressor)

#Fitting the xgboost model with all the possible variables
model_xgb <- train(target ~ venue_code + opp_code + team_code + gf_rolling_3 + ga_rolling_3 +
                   sh_rolling_3 + sot_rolling_3 + dist_rolling_3 + form_rolling_3 + xg_rolling_3 + poss_rolling_3 + 
                   sota_rolling_3 + save._rolling_3 + cs_rolling_3 + psxg_rolling_3 + cmp_rolling_3 + cmp._rolling_3 +
                   prgdist_rolling_3 + ast_rolling_3 + ppa_rolling_3 + prog_rolling_3 + sca_rolling_3 + gca_rolling_3 +
                   tklw_rolling_3 + int_rolling_3 + tkl.int_rolling_3 + err_rolling_3 + succ_rolling_3 + succ._rolling_3 +
                   crdy_rolling_3 + fls_rolling_3 + won._rolling_3, data = matches_df, method = "xgbTree",
                   trControl = trainControl("cv",number = 10), scale=T)
varImp(model_xgb)

#Fitting the logistic regression model with all the possible variables
logreg <- glm(target ~ venue_code + opp_code + team_code, gf_rolling_3 + ga_rolling_3 +
                sh_rolling_3 + sot_rolling_3 + dist_rolling_3 + form_rolling_3 + xg_rolling_3 + poss_rolling_3 + 
                sota_rolling_3 + save._rolling_3 + cs_rolling_3 + psxg_rolling_3 + cmp_rolling_3 + cmp._rolling_3 +
                prgdist_rolling_3 + ast_rolling_3 + ppa_rolling_3 + prog_rolling_3 + sca_rolling_3 + gca_rolling_3 +
                tklw_rolling_3 + int_rolling_3 + tkl.int_rolling_3 + err_rolling_3 + succ_rolling_3 + succ._rolling_3 +
                crdy_rolling_3 + fls_rolling_3 + won._rolling_3,
                family = binomial, data = matches_df)

summary(logreg)

#Fitting the CART model with all the possible variables
cartmd <- rpart(tts ~ venue_code + opp_code + team_code + gf_rolling_365 + ga_rolling_365 + form_rolling_365 + 
                xg_rolling_365 + xga_rolling_365 + poss_rolling_365 + cs_rolling_365, gf_rolling_3 + ga_rolling_3 +
                sh_rolling_3 + sot_rolling_3 + dist_rolling_3 + form_rolling_3 + xg_rolling_3 + poss_rolling_3 + 
                sota_rolling_3 + save._rolling_3 + cs_rolling_3 + psxg_rolling_3 + cmp_rolling_3 + cmp._rolling_3 +
                prgdist_rolling_3 + ast_rolling_3 + ppa_rolling_3 + prog_rolling_3 + sca_rolling_3 + gca_rolling_3 +
                tklw_rolling_3 + int_rolling_3 + tkl.int_rolling_3 + err_rolling_3 + succ_rolling_3 + succ._rolling_3 +
                crdy_rolling_3 + fls_rolling_3 + won._rolling_3,
                data = matches_df, method = 'class', control = rpart.control(minsplit = 20, cp = 0))

rpart.plot(cartmd, nn=T, main="Maximal Tree for Match Result")
summary(cartmd)
cartmd$variable.importance

print(cartmd)
printcp(cartmd)
plotcp(cartmd)

#Pruning the tree using the best cp value
cp1 <- 0.002
cartmd.pruned <- prune(cartmd, cp=cp1)
printcp(cartmd.pruned)
rpart.plot(cartmd.pruned, nn=T, main = "Pruned Tree for Match Result")
summary(cartmd.pruned)



