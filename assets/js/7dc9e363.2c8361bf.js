"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[3591],{3905:function(e,t,r){r.d(t,{Zo:function(){return u},kt:function(){return p}});var n=r(7294);function o(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function a(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function l(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?a(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function s(e,t){if(null==e)return{};var r,n,o=function(e,t){if(null==e)return{};var r,n,o={},a=Object.keys(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||(o[r]=e[r]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var i=n.createContext({}),m=function(e){var t=n.useContext(i),r=t;return e&&(r="function"==typeof e?e(t):l(l({},t),e)),r},u=function(e){var t=m(e.components);return n.createElement(i.Provider,{value:t},e.children)},c={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},b=n.forwardRef((function(e,t){var r=e.components,o=e.mdxType,a=e.originalType,i=e.parentName,u=s(e,["components","mdxType","originalType","parentName"]),b=m(r),p=o,f=b["".concat(i,".").concat(p)]||b[p]||c[p]||a;return r?n.createElement(f,l(l({ref:t},u),{},{components:r})):n.createElement(f,l({ref:t},u))}));function p(e,t){var r=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=r.length,l=new Array(a);l[0]=b;var s={};for(var i in t)hasOwnProperty.call(t,i)&&(s[i]=t[i]);s.originalType=e,s.mdxType="string"==typeof e?e:o,l[1]=s;for(var m=2;m<a;m++)l[m]=r[m];return n.createElement.apply(null,l)}return n.createElement.apply(null,r)}b.displayName="MDXCreateElement"},9614:function(e,t,r){r.r(t),r.d(t,{frontMatter:function(){return s},contentTitle:function(){return i},metadata:function(){return m},toc:function(){return u},default:function(){return b}});var n=r(7462),o=r(3366),a=(r(7294),r(3905)),l=["components"],s={},i="AutoML - Regression",m={unversionedId:"Examples/AutoML-Regression",id:"Examples/AutoML-Regression",isDocsHomePage:!1,title:"AutoML - Regression",description:"A basic regression example",source:"@site/docs/Examples/AutoML-Regression.md",sourceDirName:"Examples",slug:"/Examples/AutoML-Regression",permalink:"/FLAML/docs/Examples/AutoML-Regression",editUrl:"https://github.com/microsoft/FLAML/edit/main/website/docs/Examples/AutoML-Regression.md",tags:[],version:"current",frontMatter:{},sidebar:"docsSidebar",previous:{title:"AutoML - Rank",permalink:"/FLAML/docs/Examples/AutoML-Rank"},next:{title:"AutoML - Time Series Forecast",permalink:"/FLAML/docs/Examples/AutoML-Time series forecast"}},u=[{value:"A basic regression example",id:"a-basic-regression-example",children:[{value:"Sample output",id:"sample-output",children:[],level:4}],level:3},{value:"Multi-output regression",id:"multi-output-regression",children:[],level:3}],c={toc:u};function b(e){var t=e.components,r=(0,o.Z)(e,l);return(0,a.kt)("wrapper",(0,n.Z)({},c,r,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"automl---regression"},"AutoML - Regression"),(0,a.kt)("h3",{id:"a-basic-regression-example"},"A basic regression example"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'from flaml import AutoML\nfrom sklearn.datasets import fetch_california_housing\n\n# Initialize an AutoML instance\nautoml = AutoML()\n# Specify automl goal and constraint\nautoml_settings = {\n    "time_budget": 1,  # in seconds\n    "metric": \'r2\',\n    "task": \'regression\',\n    "log_file_name": "california.log",\n}\nX_train, y_train = fetch_california_housing(return_X_y=True)\n# Train with labeled input data\nautoml.fit(X_train=X_train, y_train=y_train,\n           **automl_settings)\n# Predict\nprint(automl.predict(X_train))\n# Print the best model\nprint(automl.model.estimator)\n')),(0,a.kt)("h4",{id:"sample-output"},"Sample output"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre"},"[flaml.automl: 11-15 07:08:19] {1485} INFO - Data split method: uniform\n[flaml.automl: 11-15 07:08:19] {1489} INFO - Evaluation method: holdout\n[flaml.automl: 11-15 07:08:19] {1540} INFO - Minimizing error metric: 1-r2\n[flaml.automl: 11-15 07:08:19] {1577} INFO - List of ML learners in AutoML Run: ['lgbm', 'rf', 'catboost', 'xgboost', 'extra_tree']\n[flaml.automl: 11-15 07:08:19] {1826} INFO - iteration 0, current learner lgbm\n[flaml.automl: 11-15 07:08:19] {1944} INFO - Estimated sufficient time budget=846s. Estimated necessary time budget=2s.\n[flaml.automl: 11-15 07:08:19] {2029} INFO -  at 0.2s,  estimator lgbm's best error=0.7393,     best estimator lgbm's best error=0.7393\n[flaml.automl: 11-15 07:08:19] {1826} INFO - iteration 1, current learner lgbm\n[flaml.automl: 11-15 07:08:19] {2029} INFO -  at 0.3s,  estimator lgbm's best error=0.7393,     best estimator lgbm's best error=0.7393\n[flaml.automl: 11-15 07:08:19] {1826} INFO - iteration 2, current learner lgbm\n[flaml.automl: 11-15 07:08:19] {2029} INFO -  at 0.3s,  estimator lgbm's best error=0.5446,     best estimator lgbm's best error=0.5446\n[flaml.automl: 11-15 07:08:19] {1826} INFO - iteration 3, current learner lgbm\n[flaml.automl: 11-15 07:08:19] {2029} INFO -  at 0.4s,  estimator lgbm's best error=0.2807,     best estimator lgbm's best error=0.2807\n[flaml.automl: 11-15 07:08:19] {1826} INFO - iteration 4, current learner lgbm\n[flaml.automl: 11-15 07:08:19] {2029} INFO -  at 0.5s,  estimator lgbm's best error=0.2712,     best estimator lgbm's best error=0.2712\n[flaml.automl: 11-15 07:08:19] {1826} INFO - iteration 5, current learner lgbm\n[flaml.automl: 11-15 07:08:19] {2029} INFO -  at 0.5s,  estimator lgbm's best error=0.2712,     best estimator lgbm's best error=0.2712\n[flaml.automl: 11-15 07:08:19] {1826} INFO - iteration 6, current learner lgbm\n[flaml.automl: 11-15 07:08:20] {2029} INFO -  at 0.6s,  estimator lgbm's best error=0.2712,     best estimator lgbm's best error=0.2712\n[flaml.automl: 11-15 07:08:20] {1826} INFO - iteration 7, current learner lgbm\n[flaml.automl: 11-15 07:08:20] {2029} INFO -  at 0.7s,  estimator lgbm's best error=0.2197,     best estimator lgbm's best error=0.2197\n[flaml.automl: 11-15 07:08:20] {1826} INFO - iteration 8, current learner xgboost\n[flaml.automl: 11-15 07:08:20] {2029} INFO -  at 0.8s,  estimator xgboost's best error=1.4958,  best estimator lgbm's best error=0.2197\n[flaml.automl: 11-15 07:08:20] {1826} INFO - iteration 9, current learner xgboost\n[flaml.automl: 11-15 07:08:20] {2029} INFO -  at 0.8s,  estimator xgboost's best error=1.4958,  best estimator lgbm's best error=0.2197\n[flaml.automl: 11-15 07:08:20] {1826} INFO - iteration 10, current learner xgboost\n[flaml.automl: 11-15 07:08:20] {2029} INFO -  at 0.9s,  estimator xgboost's best error=0.7052,  best estimator lgbm's best error=0.2197\n[flaml.automl: 11-15 07:08:20] {1826} INFO - iteration 11, current learner xgboost\n[flaml.automl: 11-15 07:08:20] {2029} INFO -  at 0.9s,  estimator xgboost's best error=0.3619,  best estimator lgbm's best error=0.2197\n[flaml.automl: 11-15 07:08:20] {1826} INFO - iteration 12, current learner xgboost\n[flaml.automl: 11-15 07:08:20] {2029} INFO -  at 0.9s,  estimator xgboost's best error=0.3619,  best estimator lgbm's best error=0.2197\n[flaml.automl: 11-15 07:08:20] {1826} INFO - iteration 13, current learner xgboost\n[flaml.automl: 11-15 07:08:20] {2029} INFO -  at 1.0s,  estimator xgboost's best error=0.3619,  best estimator lgbm's best error=0.2197\n[flaml.automl: 11-15 07:08:20] {1826} INFO - iteration 14, current learner extra_tree\n[flaml.automl: 11-15 07:08:20] {2029} INFO -  at 1.1s,  estimator extra_tree's best error=0.7197,       best estimator lgbm's best error=0.2197\n[flaml.automl: 11-15 07:08:20] {2242} INFO - retrain lgbm for 0.0s\n[flaml.automl: 11-15 07:08:20] {2247} INFO - retrained model: LGBMRegressor(colsample_bytree=0.7610534336273627,\n              learning_rate=0.41929025492645006, max_bin=255,\n              min_child_samples=4, n_estimators=45, num_leaves=4,\n              reg_alpha=0.0009765625, reg_lambda=0.009280655005879943,\n              verbose=-1)\n[flaml.automl: 11-15 07:08:20] {1608} INFO - fit succeeded\n[flaml.automl: 11-15 07:08:20] {1610} INFO - Time taken to find the best model: 0.7289648056030273\n[flaml.automl: 11-15 07:08:20] {1624} WARNING - Time taken to find the best model is 73% of the provided time budget and not all estimators' hyperparameter search converged. Consider increasing the time budget.\n")),(0,a.kt)("h3",{id:"multi-output-regression"},"Multi-output regression"),(0,a.kt)("p",null,"We can combine ",(0,a.kt)("inlineCode",{parentName:"p"},"sklearn.MultiOutputRegressor")," and ",(0,a.kt)("inlineCode",{parentName:"p"},"flaml.AutoML")," to do AutoML for multi-output regression."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'from flaml import AutoML\nfrom sklearn.datasets import make_regression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.multioutput import MultiOutputRegressor\n\n# create regression data\nX, y = make_regression(n_targets=3)\n\n# split into train and test data\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)\n\n# train the model\nmodel = MultiOutputRegressor(AutoML(task="regression", time_budget=60))\nmodel.fit(X_train, y_train)\n\n# predict\nprint(model.predict(X_test))\n')),(0,a.kt)("p",null,"It will perform AutoML for each target, each taking 60 seconds."))}b.isMDXComponent=!0}}]);