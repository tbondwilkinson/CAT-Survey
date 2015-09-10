#include <Rcpp.h>
#include <RcppGSL.h>
#include <boost/math/special_functions/hermite.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <gsl/gsl_integration.h>
#include <boost/math/distributions/normal.hpp>
#include <boost/variant.hpp>

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(BH)]]

#include <algorithm>
#include <math.h>
#include <string>
#include <vector>

using namespace Rcpp;



struct Cat {
	std::vector<double> guessing;
	std::vector<double> discrimination;
	std::vector<double> prior_values;
	std::string prior_name;
	std::vector<double> prior_params;
	std::vector<int> answers;
	double D;
	// X is the abcissa values for whatever mode of integation
	// in trapezoidal, this doesn't matter that much
	// in hermite-gauss, it should be the roots of the hermite polynomial
	std::vector<double> X;
	std::vector<double> theta_est;
	std::vector<std::vector<double> > difficulty;
	std::vector<int> applicable_rows;
	bool poly;
	enum IntegrationType {
		TRAPEZOID, HERMITE, QAG
	};
	IntegrationType integration_method;
	boost::variant<boost::math::normal_distribution<double>,
		boost::math::cauchy_distribution<double>
    	boost::math::students_t_distribution<double> > distribution;

    struct DistributionVisitor : public boost::static_visitor<double> {
    	double operator()(const boost::math::normal_distribution<double> normal) const { return boost::math::pdf(normal, x); }
    	double operator()(const boost::math::cauchy_distribution<double> cauchy) const { return boost::math::pdf(cauchy, x); }
    	double operator()(const boost::math::students_t_distribution<double> t) const { 
    		return 1.0 / prior_params[1] * boost::math::pdf(t, (x - prior_params[0]) / prior_params[1]);
    	}
    	double x;
	};

	Cat(std::vector<double> guess, std::vector<double> disc, std::vector<double> pri_v, std::string pri_n,
		std::vector<double> pri_p, std::vector<int> ans, double d, std::vector<double> x, std::vector<double> t_est,
		std::vector<std::vector<double> > diff, std::vector<int> app_rows, bool p, std::string im) :
		guessing(guess), discrimination(disc), prior_values(pri_v), prior_name(pri_n), prior_params(pri_p),
		answers(ans), D(d), X(x), theta_est(t_est), difficulty(diff), applicable_rows(app_rows), poly(p),
		{
			if (im.compare("qag") == 0) {
				integration_method = QAG;
				if (prior_name.compare("normal") == 0) {
					distribution = boost::math::normal_distribution<double>(prior_params[0], prior_params[1]);
				} else if (prior_name.compare("dcauchy") == 0) {
					distribution = boost::math::cauchy_distribution<double>(prior_params[0], prior_params[1]);
				} else if (prior_name.compare("t") == 0) {
					distribution = boost::math::students_t_distribution<double>(prior_params[2]);
				} else {
					return NumericVector();
				}
			} else if (im.compare("hermite") == 0) {
				integration_methd = HERMITE;
			} else {
				integration_method = TRAPEZOID;
			}
		}};

	double prior(double x) {
		DistributionVisitor visitor;
		visitor.x = x;
		return boost::apply_visitor(visitor, distribution);
	}
};

double trapezoidal_integration(std::vector<double>& x, std::vector<double>& fx) {
	double val = 0;
	for (unsigned int i = 0; i < x.size() - 1; ++i) {
		val += (x[i + 1] - x[i]) * (fx[i + 1] + fx[i]) / 2.0;
	}
	return val;
}

double integrateTopTheta (double x, void * params) {
  	Cat cat = *(Cat *) params;
  	return x * likelihood(cat, x, cat.applicable_rows) * cat.prior(x));
}

struct TopParams {
	Cat cat;
	double theta_hat;
};

double integrateTopVar (double x, void * params) {
	TopParams top_params = *(TopParams *) params;
	Cat cat = top_params.cat;    
	double theta_hat = top_params.theta_hat;
	return (x - theta_hat) * (x - theta_hat) * likelihood(cat, x, cat.applicable_rows) * cat.prior(x);
}

double integrateBottom (double x, void * params) {
  	Cat cat = *(Cat *) params;
  	return likelihood(cat, x, cat.applicable_rows) * cat.prior(x));
}

double gsl_integrate(gsl_function F) {
	gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
  
  	double result, error;

	gsl_integration_qag (&F, -5.0, 5.0, 0, 1e-7, 1000, w, &result, &error); 
	gsl_integration_workspace_free (w);

	return result;
}

// This method is using Rcpp syntactic sugar (literally what they call it)
// instead of C++ because of dnorm/dcauchy/dt functions that we need exposed.
// These are optimized (probably) anyways so it's not like using a C++ version of these functions
// would help us anways.
// And this function is only called once per estimation.
NumericVector prior(NumericVector& values, CharacterVector& name, NumericVector& params) {
	std::string str_name = as<std::string>(name[0]);
	if (str_name.compare("normal") == 0) {
		return dnorm(values, params[0], params[1]);
	} else if (str_name.compare("dcauchy") == 0) {
		return dcauchy(values, params[0], params[1]);
	} else if (str_name.compare("t") == 0) {
		return 1.0 / params[1] * dt((values - params[0]) / params[1], params[2]);
	} else {
		return NumericVector();
	}
}

void three_pl(Cat& cat, double theta, int question, std::vector<double>& ret_prob) {
	unsigned int diff_size = cat.difficulty[question].size();
	double D = cat.D;
	double discrimination = cat.discrimination[question];
	double guessing = cat.guessing[question];
	for (unsigned int i = 0; i < diff_size; ++i) {
		double exp_prob = exp(D * discrimination * (theta - cat.difficulty[question][i]));
		ret_prob.push_back(guessing + (1 - guessing) * (exp_prob) / (1 + exp_prob));
	}
}

double likelihood(Cat & cat, double theta, std::vector<int> items) {
	if (cat.poly) {
		double L = 1.0;
		for (unsigned int i = 0; i < items.size(); ++i) {
			int question = items[i];
			std::vector<double> question_cdf;
			question_cdf.push_back(1.0);
			three_pl(cat, theta, question, question_cdf);
			question_cdf.push_back(0.0);

			std::vector<double> question_pdf;
			for (unsigned int j = 0; j < question_cdf.size() - 1; ++j) {
				question_pdf.push_back(question_cdf[j] - question_cdf[j + 1]);
			}
			L *= question_pdf[cat.answers[question] - 1];
		}
		return L;
	} else {
		// Non-poly case not implemented.
		return -1;
	}
}

double estimateTheta(Cat & cat) {
	double results = 0.0;
	if (cat.integration_method == Cat::TRAPEZOID) {
		std::vector<double> fx;
		std::vector<double> fx_x;
		for (unsigned int i = 0; i < cat.X.size(); ++i) {
			fx.push_back(likelihood(cat, cat.X[i], cat.applicable_rows) * cat.prior_values[i]);
			fx_x.push_back(cat.X[i] * fx[i]);
		}
		results = trapezoidal_integration(cat.X, fx_x) / trapezoidal_integration(cat.X, fx);
	} else if (cat.integration_method == Cat::QAG) {
		gsl_function Ftop;
		Ftop.function = &integrateTopTheta;
		Ftop.params = &cat;

		gsl_function Fbottom;
		Fbottom.function = &integrateBottom;
		Fbottom.params = &cat;

		results = gsl_integration_qag(Ftop) / gsl_integration_qag(Fbottom);
	}
	return results;
}

double estimateSE(Cat & cat, double theta_hat) {
	double results = 0.0;
	if (cat.integration_method == Cat::TRAPEZOID) {
		std::vector<double> fx;
		std::vector<double> fx_theta;
		for (unsigned int i = 0; i < cat.X.size(); ++i) {
			fx.push_back(likelihood(cat, cat.X[i], cat.applicable_rows) * cat.prior_values[i]);
			fx_theta.push_back((cat.X[i] - theta_hat) * (cat.X[i] - theta_hat) * fx[i]);
		}
		results = sqrt(trapezoidal_integration(cat.X, fx_theta) / trapezoidal_integration(cat.X, fx));
	} else if (cat.integration_method == Cat::QAG) {
		gsl_function Ftop;
		Ftop.function = &integrateTopVar;
		TopParams top_params;
		top_params.cat = cat;
		top_params.theta_hat = theta_hat;
		Ftop.params = &top_params;

		results = sqrt(gsl_integration_qag(Ftop) / gsl_integration_qag(Fbottom));
	}

	return results;
}

double expectedPV(Cat cat, int item) {
	if (cat.poly) {
		std::vector<double> variances;
		cat.applicable_rows.push_back(item);
		for (unsigned int i = 0, size = cat.difficulty[item].size() + 1; i < size; ++i) {
			cat.answers[item] = i + 1;
			variances.push_back(estimateSE(cat, estimateTheta(cat)));
			variances[i] *= variances[i];
		}
		cat.answers[item] = NA_INTEGER;
		cat.applicable_rows.pop_back();
		std::vector<double> question_cdf;
		question_cdf.push_back(1.0);
		three_pl(cat, estimateTheta(cat), item, question_cdf);
		question_cdf.push_back(0.0);
		double sum = 0.0;
		for (unsigned int i = 0, size = question_cdf.size() - 1; i < size; ++i) {
			sum += variances[i] * (question_cdf[i] - question_cdf[i + 1]);
		}
		return sum;
	} else {
		// not implemented
		return -1;
	}
}

// [[Rcpp::export]]
List nextItemEPVcpp(S4 cat_df) {
	// Precalculate the priors, since they never change given a cat object.
	NumericVector X = cat_df.slot("X");
	CharacterVector priorName = cat_df.slot("priorName");
	NumericVector priorParams = cat_df.slot("priorParams");
	std::vector<double> prior_values = as<std::vector<double> >(prior(X, priorName, priorParams));

	// Precalculate the rows that have been answered.
	std::vector<int> applicable_rows;
	std::vector<int> nonapplicable_rows;
	std::vector<int> answers = as<std::vector<int> >(cat_df.slot("answers"));
	for (int i = 0; i < answers.size(); i++) {
		if (answers[i] != NA_INTEGER) {
			applicable_rows.push_back(i);
		} else {
			nonapplicable_rows.push_back(i + 1);
		}
	}
	// Unpack the difficulty list
	std::vector<std::vector<double> > difficulty;
	List cat_difficulty = cat_df.slot("difficulty");
	for (List::iterator itr = cat_difficulty.begin(); itr != cat_difficulty.end(); ++itr) {
		difficulty.push_back(as<std::vector<double> >(*itr));
	}
	// Construct the c++ cat object
	Cat cat(as<std::vector<double> >(cat_df.slot("guessing")), as<std::vector<double> >(cat_df.slot("discrimination")),
		prior_values, as<std::string>(priorName), as<std::vector<double> >(priorParams),
		as<std::vector<int> >(cat_df.slot("answers")), as<std::vector<double> >(cat_df.slot("D"))[0],
		as<std::vector<double> >(cat_df.slot("X")), as<std::vector<double> >(cat_df.slot("Theta.est")),
		difficulty, applicable_rows, as<std::vector<bool> >(cat_df.slot("poly"))[0], as<sd::string>(cat_df.slot("integration")));
	// For every unanswered item, calculate the epv of that item
	std::vector<double> epvs;
	int min_item = -1;
	double min_epv = std::numeric_limits<double>::max();
	for (unsigned int i = 0, size = nonapplicable_rows.size(); i < size; ++i) {
		epvs.push_back(expectedPV(cat, nonapplicable_rows[i] - 1));
		if (epvs[i] < min_epv) {
			min_item = nonapplicable_rows[i];
			min_epv = epvs[i];
		}
	}
	DataFrame all_estimates = DataFrame::create(Named("questions")=nonapplicable_rows, Named("EPV")=epvs);
	NumericVector next_item = wrap(min_item);
	return List::create(Named("all.estimates")=all_estimates, Named("next.item")=next_item);
}

// [[Rcpp::export]]
List lookAheadEPVcpp(S4 cat_df, NumericVector item) {
	int look_ahead_item = as<int>(item) - 1;
	NumericVector X = cat_df.slot("X");
	CharacterVector priorName = cat_df.slot("priorName");
	NumericVector priorParams = cat_df.slot("priorParams");
	std::vector<double> prior_values = as<std::vector<double> >(prior(X, priorName, priorParams));

	// Precalculate the rows that have been answered.
	std::vector<int> applicable_rows;
	std::vector<int> nonapplicable_rows;
	std::vector<int> answers = as<std::vector<int> >(cat_df.slot("answers"));
	for (int i = 0; i < answers.size(); i++) {
		if (i == look_ahead_item) {
			applicable_rows.push_back(i);
		} else if (answers[i] != NA_INTEGER) {
			applicable_rows.push_back(i);
		} else {
			nonapplicable_rows.push_back(i + 1);
		}
	}
	// Unpack the difficulty list
	std::vector<std::vector<double> > difficulty;
	List cat_difficulty = cat_df.slot("difficulty");
	for (List::iterator itr = cat_difficulty.begin(); itr != cat_difficulty.end(); ++itr) {
		difficulty.push_back(as<std::vector<double> >(*itr));
	}
	// Construct the c++ cat object
	Cat cat(as<std::vector<double> >(cat_df.slot("guessing")), as<std::vector<double> >(cat_df.slot("discrimination")),
		prior_values, as<std::string>(priorName), as<std::vector<double> >(priorParams),
		as<std::vector<int> >(cat_df.slot("answers")), as<std::vector<double> >(cat_df.slot("D"))[0],
		as<std::vector<double> >(cat_df.slot("X")), as<std::vector<double> >(cat_df.slot("Theta.est")),
		difficulty, applicable_rows, as<std::vector<bool> >(cat_df.slot("poly"))[0], as<sd::string>(cat_df.slot("integration")));
	if (look_ahead_item >= cat.answers.size()) {
		stop("Item out of bounds");
	} else if (!(cat.answers[look_ahead_item] == NA_INTEGER)) {
		stop("Item already answered");
	}
	std::vector<DataFrame> all_epvs;
	std::vector<double> min_items;
	for (unsigned int answer = 1, size = cat.difficulty[look_ahead_item].size() + 2; answer < size; ++answer) {
		cat.answers[look_ahead_item] = answer;
		std::vector<double> epvs;
		int min_item = -1;
		double min_epv = std::numeric_limits<double>::max();
		for (unsigned int i = 0, size = nonapplicable_rows.size(); i < size; ++i) {
			epvs.push_back(expectedPV(cat, nonapplicable_rows[i] - 1));
			if (epvs[i] < min_epv) {
				min_item = nonapplicable_rows[i];
				min_epv = epvs[i];
			}
		}
		all_epvs.push_back(DataFrame::create(Named("questions")=nonapplicable_rows, Named("epvs")=epvs));
		min_items.push_back(min_item);
	}
	return List::create(Named("all.epvs")=all_epvs, Named("next.items")=min_items);
}

