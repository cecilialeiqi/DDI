#include <algorithm>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <iomanip>
#include <vector>
#include<Eigen/Dense>
#include<Eigen/Sparse>
#include <unordered_set>
#include <set>
#include <fstream>
using namespace Eigen;
#pragma GCC diagnostic ignored "-Wwrite-strings"
using namespace std;
typedef Eigen::SparseMatrix<double,RowMajor> SpMat;
typedef Eigen::Triplet<double> T;

double max(double a, double b)
{
	return a>b?a:b;
}
void read(vector<unordered_set<int>> & Y_index, SpMat & Drug, int k, vector<pair<int,int>> & train, vector<pair<int,int>> & test, int train_size, int test_size)  //645*645
{
	clock_t st=clock();
    vector<T> Drug_in_triplets;
    ifstream ifile("index_features.csv");
    int i,j,v;
	int num=0;
    while (!ifile.eof()){
	ifile>>i;
	ifile>>j;
	num++;
        Drug_in_triplets.push_back(T(i,j,1));
    if (num%10000==0)
		cout<<num<<':'<<i<<','<<j<<endl;
	}
    Drug.setFromTriplets(Drug_in_triplets.begin(),Drug_in_triplets.end());
    ifstream ifile2("input_matrix.csv");
	vector<bool> drug_pair(645*645,false);
	vector<int> nDDI(645*645,0);
    num=0;
	int nnz=0;
	int total=0;
	while (!ifile2.eof()){
        ifile2>>i;
        ifile2>>j;
        ifile2>>v;
		if (!drug_pair[i*645+j]){
        	total+=2;
			drug_pair[i*645+j]=true;
			drug_pair[j*645+i]=true;
		}
		nDDI[i*645+j]++;
		nnz++;
        if (v>=k)
            continue;
        Y_index[i*645+j].insert(v);
	    Y_index[j*645+i].insert(v);
		num++;
    	if (num%100000==0)
			cout<<num<<':'<<i<<','<<j<<','<<v<<endl;
	}
    if (train_size>total){
        train_size=total*0.9;
    	test_size=total-train_size;
    }
	cout<<train_size<<','<<test_size<<endl;
    for (int i=0;i<645*645;i++){
		if (!drug_pair[i])
			continue;
		int r=rand()%total;
	if (r<train_size)
	    train.push_back(make_pair(i/645,i%645));
	else if (r>=train_size && r<train_size+test_size)
	    test.push_back(make_pair(i/645,i%645));
    }
	cout<<train.size()<<','<<test.size()<<endl;
	cout<<"Reading elapsed time:"<<(clock()-st)*1.0/CLOCKS_PER_SEC<<endl;
}

void initialize(MatrixXd & O_init, MatrixXd & Q_init, double & b_init){
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0,1);
    auto gaussian=[&](double) {return distribution(generator);};
    O_init=MatrixXd::NullaryExpr(O_init.rows(),O_init.cols(),gaussian)*0.1;
    Q_init=MatrixXd::NullaryExpr(Q_init.rows(),Q_init.cols(),gaussian)*0.01;
    b_init=0.0;
	cout<<"Finished initialization"<<endl;
}

//(DrugO, Q_init,b_init,Drug,Y_index,K,train, grad, J_val);

double dJO(MatrixXd & DrugO, MatrixXd & DrugQ,double b, SpMat & Drug,vector<unordered_set<int>> & Y_index, int K, vector<pair<int,int>> & train, MatrixXd & grad){
    //gradient for O
	double J_val=0;
    grad.setZero(Drug.cols(),DrugO.cols());
    J_val=0;
    for (int counter=0; counter<train.size(); counter++){
        int i= train[counter].first;
        int j= train[counter].second;
        for (int k=0;k<K;k++){
            VectorXd tmp=DrugO.row(j).cwiseProduct(DrugQ.row(k));
            double f_val=DrugO.row(i).dot(tmp)+b;
            //grad=grad+(1/(1+exp(-f_val))-Y(i,j,k))* (Drug(i,:)'*tmp);
            double val=1/(1+exp(-f_val));
	    	if (Y_index[i*645+j].find(k)==Y_index[i*645+j].end()){
	        	J_val+=log(1+exp(f_val));
	    	}else{
	        	J_val+=-f_val+log(1+exp(f_val));
				val-=1;
	    	}
	    	for (SparseMatrix<double,RowMajor>::InnerIterator it(Drug,i); it; ++it)
	    	{
	        	grad.row(it.col())+=val*tmp;
	   	 	}
        }
    }
    J_val/=train.size();
    grad*=2.0/train.size();
	return J_val;
}

double functionJ(MatrixXd & DrugO, MatrixXd & DrugQ, double b, vector<unordered_set<int>> & Y_index, vector<pair<int,int>> & train, int K){
    	//O_init=O_pre;
    double J_val=0;
    for (int counter=0;counter<train.size();counter++){
        int i= train[counter].first;
        int j= train[counter].second;
		VectorXd tmp=DrugO.row(i).cwiseProduct(DrugO.row(j));
        for (int k=0;k<K;k++){
            double f_val=tmp.dot(DrugQ.row(k))+b;
            if (Y_index[i*645+j].find(k)==Y_index[i*645+j].end())
	            J_val+=log(1+exp(f_val));
	        else
	            J_val+=-f_val+log(1+exp(f_val));
        }
    }
    J_val/=train.size();
    return J_val;
}

//MatrixXd O_pre=fO(Drug,Y_index,O_init,Q_init,b_init,beta,gamma,train,tolerance,lambda,k);

void fO(SpMat & Drug, vector<unordered_set<int>> & Y_index, MatrixXd O, MatrixXd & Q_init, double b_init, double beta, double gamma, vector<pair<int,int>> & train, double tolerance, double lambda, int K, MatrixXd & O_new){
    double theta=1;
	double J,J_hat;
    clock_t st=clock();
    for (int t=0;t<100;t++){
        MatrixXd O_old=O_new;
        MatrixXd DrugO=Drug*O_old;
        //compute the gradient,and store the J's value for next steps
        double J_val;
	    MatrixXd grad(O.rows(),O.cols());
	    J_val=dJO(DrugO, Q_init,b_init,Drug,Y_index,K,train, grad);
            while(gamma>=1e-6){
                //gradient descent
                O_new=O-gamma*grad;
                O_new=(1/(1+lambda*gamma))*O_new;
                //boundary condition
                J_hat=J_val+(grad.transpose()*(O_new-O)).trace()+pow((O_new-O).norm(),2)/(2*gamma);
                DrugO=Drug*O_new;
				J=functionJ(DrugO,Q_init,b_init,Y_index,train,K);
				if (J<=J_hat)
                    break;
                gamma=gamma*beta;
            }
        double error=(O-O_new).norm()/max(1,O_new.norm());
       printf("O:iter num %d, norm(tGk): %1.2e, step-size: %1.2e, cost: %1.5e, elapsed_time: %1.2e\n",t,error,gamma,J,(clock()-st)*1.0/CLOCKS_PER_SEC);
        st=clock();
        //accelerate part
        theta = 2.0/(1 + sqrt(1+4/(theta*theta)));
        if (((O-O_new).transpose()*(O_new-O_old)).trace()>0){
            O_new = O_old;
            O = O_new;
            theta = 1;
            gamma=1;
		}
        else
            O = O_new + (1-theta)*(O_new-O_old);
        //gamma=1;
        if (error<tolerance)
            break;
	}
	O_new=O;
	//return O_new;
}


double dJQ(MatrixXd & DrugO, MatrixXd & DrugQ,double b, vector<unordered_set<int>> & Y_index, int K, vector<pair<int,int>> & train, MatrixXd & grad){
    //gradient for O
	double J_val=0;
    grad.setZero(DrugQ.rows(),DrugQ.cols());
    J_val=0;
    for (int counter=0; counter<train.size(); counter++){
        int i= train[counter].first;
        int j= train[counter].second;
        VectorXd tmp=DrugO.row(i).cwiseProduct(DrugO.row(j));
        for (int k=0;k<K;k++){
            double f_val=DrugQ.row(k).dot(tmp)+b;
            //grad=grad+(1/(1+exp(-f_val))-Y(i,j,k))* (Drug(i,:)'*tmp);
            double val=1/(1+exp(-f_val));
	    	if (Y_index[i*645+j].find(k)==Y_index[i*645+j].end()){
	        	J_val+=log(1+exp(f_val));
	    	}else{
	        	J_val+=-f_val+log(1+exp(f_val));
				val-=1;
	    	}
	        grad.row(k)+=val*tmp;
        }
    }
    J_val/=train.size();
    grad/=train.size();
	return J_val;
}


void fQ(vector<unordered_set<int>> & Y_index, MatrixXd & DrugO, MatrixXd Q, double b_init, double beta, double gamma, vector<pair<int,int>> & train, double tolerance, double lambda, int K, MatrixXd & Q_new){
    double theta=1;
	double J,J_hat;
    clock_t st=clock();
    for (int t=0;t<100;t++){
        MatrixXd Q_old=Q_new;
        //compute the gradient,and store the J's value for next steps
        double J_val;
		MatrixXd grad(Q.rows(),Q.cols());
	    J_val=dJQ(DrugO, Q_old,b_init, Y_index,K,train, grad);
            while(gamma>=1e-6){
                //gradient descent
                Q_new=Q-gamma*grad;
                Q_new=(1/(1+lambda*gamma))*Q_new;
                //boundary condition
				J_hat=J_val+(grad.transpose()*(Q_new-Q)).trace()+pow((Q_new-Q).norm(),2)/(2*gamma);
				J=functionJ(DrugO,Q_new,b_init,Y_index,train,K);
				if (J<=J_hat)
                    break;
                gamma=gamma*beta;
            }
        double error=(Q-Q_new).norm()/max(1,Q_new.norm());
          printf("Q:iter num %d, norm(tGk): %1.2e, step-size: %1.2e, cost: %1.5e, elapsed_time: %1.2e\n",t,error,gamma,J,(clock()-st)*1.0/CLOCKS_PER_SEC);
        st=clock();
        //accelerate part
        theta = 2.0/(1 + sqrt(1+4/(theta*theta)));
        if (((Q-Q_new).transpose()*(Q_new-Q_old)).trace()>0){
            Q_new = Q_old;
            Q = Q_new;
            theta = 1;
            gamma=10;
		}
        else
            Q = Q_new + (1-theta)*(Q_new-Q_old);
        //gamma=10;
    if (error<tolerance)
            break;
      }
	Q_new=Q;
	//return O_new;
}


double dJb(MatrixXd & DrugO, MatrixXd & DrugQ,double b, vector<unordered_set<int>> & Y_index, int K, vector<pair<int,int>> & train, double & grad){
    //gradient for O
	double J_val=0;
    grad=0;
    J_val=0;
    for (int counter=0; counter<train.size(); counter++){
        int i= train[counter].first;
        int j= train[counter].second;
        VectorXd tmp=DrugO.row(i).cwiseProduct(DrugO.row(j));
        for (int k=0;k<K;k++){
            double f_val=DrugQ.row(k).dot(tmp)+b;
            //grad=grad+(1/(1+exp(-f_val))-Y(i,j,k))* (Drug(i,:)'*tmp);
            double val=1/(1+exp(-f_val));
	    	if (Y_index[i*645+j].find(k)==Y_index[i*645+j].end()){
	        	J_val+=log(1+exp(f_val));
	    	}else{
	        	J_val+=-f_val+log(1+exp(f_val));
				val-=1;
	    	}
	        grad+=val;
        }
    }
    J_val/=train.size();
    grad/=train.size();
	return J_val;
}


double fb(vector<unordered_set<int>> & Y_index, MatrixXd & DrugO, MatrixXd & DrugQ, double b, double beta, double gamma, vector<pair<int,int>> & train, double tolerance, double lambda, int K){
    double theta=1;
	double J,J_hat;
    clock_t st=clock();
	double b_new=b;
    for (int t=0;t<100;t++){
        double b_old=b_new;
        //compute the gradient,and store the J's value for next steps
        double J_val;
		double grad;
	    J_val=dJb(DrugO,DrugQ,b_old, Y_index,K,train, grad);
        while(gamma>=1e-6){
                //gradient descent
                b_new=b-gamma*grad;
                //b_new=(1/(1+lambda*gamma))*b_new;
                //boundary condition
				J_hat=J_val;//+grad*(b_new-b)+pow(b_new-b,2)/(2*gamma);
				J=functionJ(DrugO,DrugQ,b_new,Y_index,train,K);
				if (J<=J_hat)
                    break;
                gamma=gamma*beta;
            }
        double error=abs(b-b_new)/max(1,abs(b_new));
       printf("b:iter num %d, norm(tGk): %1.2e, step-size: %1.2e, cost: %1.5e, b: %1.3e, elapsed_time: %1.2e\n",t,error,gamma,J,b_new, (clock()-st)*1.0/CLOCKS_PER_SEC);
        st=clock();
        //accelerate part
        theta = 2.0/(1 + sqrt(1+4/(theta*theta)));
        if ((b-b_new)*(b_new-b_old)>0){
            b_new = b_old;
            b = b_new;
            theta = 1;
            gamma=1;
		}
        else
            b = b_new + (1-theta)*(b_new-b_old);
        //gamma=1;
	    if (error<tolerance)
            break;
 }
	return b;
	//return O_new;
}


void model(int n, int k, int r,   SpMat & Drug,   vector<unordered_set<int>> & Y_index, vector<pair<int,int>> & train, vector<pair<int,int>> & test){
    MatrixXd O_init(881,r);
    MatrixXd Q_init(k,r);
    double b_init;
    initialize(O_init,Q_init,b_init);
    MatrixXd DrugO;
    // parameter used in the gradient descent
    int Maxiter=20;
    double beta=0.5;
    double gamma=1;
    double lambda=1e-5;
    double tolerance=1e-5;
    for (int t=0;t<Maxiter;t++){
		clock_t st=clock();
 		MatrixXd O_pre(O_init);
		fO(Drug,Y_index,O_init,Q_init,b_init,beta,gamma,train,tolerance,lambda,k,O_pre);
		double errorO= (O_init-O_pre).norm()/max(1,O_pre.norm());
    	O_init=O_pre;
    	DrugO=Drug*O_init;
    	MatrixXd Q_pre(Q_init);
		fQ(Y_index,DrugO,Q_init,b_init,beta,10.0,train,tolerance,lambda,k,Q_pre);
    	double errorQ= (Q_init-Q_pre).norm()/max(1,Q_pre.norm());
    	Q_init=Q_pre;
    	double b_pre= fb(Y_index,DrugO,Q_init,b_init,beta,gamma,train,tolerance,lambda,k);
    	double errorb= abs(b_init-b_pre)/max(1,abs(b_pre));
    	b_init=b_pre;
    	//double b_pre=0;
		//double errorb=0;
		printf("Alternate: iter num %d, norm(O): %1.2e, norm(Q): %1.2e, b: %1.3e, elapsed time: %1.2e\n", t, O_init.norm(),Q_init.norm(),b_init, clock()-st);
		if (errorO<tolerance && errorQ<tolerance && errorb<tolerance)
			break;
	}
	ofstream ofile("test.csv");
	for (int i=0;i<test.size();i++){
		ofile<<test[i].first<<','<<test[i].second<<endl;
	}
	ofile.close();
	ofstream ofile_O("O.csv");
	for (int i=0;i<O_init.rows();i++){
		ofile_O<<O_init(i,0);
		for (int j=1;j<O_init.cols();j++)
			ofile_O<<','<<O_init(i,j);
		ofile_O<<endl;
	}
	ofile_O.close();
	ofstream ofile_Q("Q.csv");
	for (int i=0;i<Q_init.rows();i++){
		ofile_Q<<Q_init(i,0);
		for (int j=1;j<Q_init.cols();j++)
			ofile_Q<<','<<Q_init(i,j);
		ofile_Q<<endl;
	}
	ofile_Q.close();
	ofstream ofile_b("b.csv");
	ofile_b<<b_init<<endl;
	ofile_b.close();
}

int main(int argc, char* argv[]){
    int n=645;
    int k=1318;
    int r=8;
	int train_size=20000;
	int test_size=2000;
    if (argc==5){
        k=atoi(argv[1]);
        r=atoi(argv[2]);
		train_size=atoi(argv[3]);
		test_size=atoi(argv[4]);
    }
	cout<<"run <k> <r> <train_size> <test_size>"<<endl;
    vector<unordered_set<int>> Y_index(n*n,unordered_set<int>());
    SpMat Drug(n,881);
    vector<pair<int,int>> train_set;
    vector<pair<int,int>> test_set;
    read(Y_index,Drug,k,train_set,test_set,train_size,test_size);
    model(n,k,r,Drug,Y_index,train_set,test_set);
}
