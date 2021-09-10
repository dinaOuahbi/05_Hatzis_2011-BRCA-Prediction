def cross_val_predictProba_10cv(clf, X, y, sample_weight, cv):
	"""This function performs standard 10CV.

	@type	clf: machine learning algorithm object
	@param	clf: The machine learning algorithm to use to fit the data.
	@type	X: array-like
	@param	X: The data to fit.
	@type	y: array-like
	@param	y: Original sample labels.
	@type	sample_weight: bool
	@param	sample_weight: Whether to apply sample weighting.
	@type	cv: cross-validation generator, LeaveOneOut object
	@param	cv: 10CV generator.

	@return: Predictions and class probabilities.
	"""
	np.random.seed(0)
	df_pred = pd.DataFrame()
	y_pred = np.array([])
	df_proba = pd.DataFrame()
	y_proba = np.empty((0, 2))
	cv_splits = cv.split(X,y)

	o = 1

	for train_index, test_index in cv_splits:
		Xtrain = X[train_index]
		Xtest = X[test_index]

		ytrain = y[train_index]
		ytest = y[test_index]

		cclf = clone(clf)

		try:
			if sample_weight:
				if not isinstance(cclf, Pipeline):
					cclf.fit(Xtrain, ytrain, sample_weight = compute_sample_weight("balanced", ytrain))
				else:
					cclf.fit(Xtrain, ytrain, model__sample_weight = compute_sample_weight("balanced", ytrain))
			else:
				cclf.fit(Xtrain, ytrain)

		except LightGBMError as err:
			print(err)
			if isinstance(cclf, Pipeline):
				adjusted_min_child_samples = int(round(Xtrain.shape[0]/float(cclf.named_steps['model'].num_leaves)))
				cclf.named_steps['model'].min_child_samples = adjusted_min_child_samples
			else:
				adjusted_min_child_samples = int(round(Xtrain.shape[0]/float(cclf.num_leaves)))
				cclf.min_child_samples = adjusted_min_child_samples

			if sample_weight:
				if not isinstance(cclf, Pipeline):
					cclf.fit(Xtrain, ytrain, sample_weight = compute_sample_weight("balanced", ytrain))
				else:
					cclf.fit(Xtrain, ytrain, model__sample_weight = compute_sample_weight("balanced", ytrain))
			else:
				cclf.fit(Xtrain, ytrain)

		if df_proba.empty:
			df_proba = pd.DataFrame(cclf.predict_proba(Xtest))
			df_proba['ind'] = test_index
		else:
			current_proba = pd.DataFrame(cclf.predict_proba(Xtest))
			current_proba['ind'] = test_index
			df_proba = pd.concat([df_proba, current_proba])

		if df_pred.empty:
			df_pred = pd.DataFrame({'ind':test_index, 'pred':cclf.predict(Xtest)})
		else:
			current_pred = pd.DataFrame({'ind':test_index, 'pred':cclf.predict(Xtest)})
			df_pred = pd.concat([df_pred, current_pred])

		print(str(o) + " predictions out of " + str(X.shape[0]))
		o = o + 1

	#print(df_proba)
	df_proba = df_proba.sort_values('ind')
	#print(df_proba)
	y_proba = df_proba.drop(['ind'], axis = 1).values

	#print(df_pred)
	df_pred = df_pred.sort_values('ind')
	#print(df_pred)
	y_pred = df_pred['pred'].values

	return y_pred, y_proba, cclf
