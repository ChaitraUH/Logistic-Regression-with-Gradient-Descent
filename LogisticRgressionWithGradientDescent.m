%calling function  
function [thetaopt,prediction,losses,epochs]= LogisticQuestion1()
     D = csvread("TrainFeaturesupdated.csv",1,2);
     trLb = csvread("TrainLabelsupdated.csv",1,2);
     %Z = ( D - mean(D(:)) ) / std(D(:));
     normD=normalize(D);
     X=normD';
     columns=size(X,2);
     onevector=ones(1,columns);
     X=[X;onevector];
     y=trLb;
     %setting values as in the question 
     m=16;
     max_epoch =1000;
     eta0=0.1;
     eta1=1;
     delta=0.00001;
     k=4;
     [thetaopt,losses,epochs]=gradientdescent(X,y,m,eta0,eta1,max_epoch,delta,k);
     plot(epochs,losses);
     title("L(\theta) by epoch");
     xlabel("epoch");
     ylabel("L(\theta)");
     %disp(thetaopt);
     Testdata = csvread("TestFeatures2.csv",1,1);
     normD1=normalize(Testdata);
     TestX=normD1';
     TestColumns=size(TestX,2);
     onevector=ones(1,TestColumns);
     TestX=[TestX;onevector];
     %prediction on test data based on optimized theta 
     prediction=predict(TestX,thetaopt,k);
end
 function [thetaopt,losses,epochs]=gradientdescent(X,y,m,eta0,eta1,max_epoch,delta,k)
    columns=size(X,2);
    rows=size(X,1);
    theta=ones(rows,k-1);
    lossold=100;
    %for each epoch
    for epoch=1:max_epoch
        disp(epoch);
        eta=eta0/(eta1+epoch);
        i1toin=columns;
        permutations = randperm(i1toin);
        %dividing into batches 
        cells=num2cell(reshape(permutations, m, (columns/m)), 1);
        sumloss=0;
        %for each batch
        for g=1:numel(cells)
            perms=cells{g};
            Lossofgradient=zeros(rows,k-1);
            for i=1:numel(perms)
                perm=perms(i);
                xi=X(:,perm);
                yi=y(perm);
                probs=findProbs(xi,theta,k);
                Lossofgradient=Lossofgradient+GradientofLoss(yi,xi,probs,k);
                sumloss=sumloss+logprob(probs,yi,k);
            end
            gradient=(-1/(m)).*Lossofgradient;
            theta=theta-(eta*(gradient));
            thetaopt=theta;
        end
        lossnew=lossfunction(sumloss,columns);
        if(lossnew>(1-delta)*lossold)
            disp("insideif");
             break;
        end
        lossold=lossnew;
        losses(epoch)=lossnew;
        epochs(epoch)=epoch;
         
    end
 end

 function [gradientl]=GradientofLoss(y,X,probs,k)
       grad=zeros(size(X,1),k-1);
       for j=1:k-1
           if(j==y)
               val=1;
           else
               val=0;
           end
           grad(:,j)=grad(:,j)+((val-probs(j))*X);
       end
       gradientl=grad;
 end
 
 %function to find the loss on each data point 
 function [losseach]=logprob(probs,y,k)
      for j=1:k
          if(j==y)
              losseach=log(probs(j));
          else
              losseach=0;
          end       
      end
 end
 
 %functon to find loss based on the sum of log loss given 
 function[loss]=lossfunction(sumlogloss,n)
     loss=(-1/n)*sumlogloss;
 end
 
 %function to find probability 
 function [probs]=findProbs(X,Q,k)
     denom=1+sum(exp(Q'*X));
     for j=1:k-1
         numer=sum(exp((Q(:,j))'*X));
         p(j)=numer/denom;
     end
     p(k)=1/denom;
     probs=p;
 end
 
 %function for prediction 
 function [prediction]=predict(X,thetaopt,k)
    columns=size(X,2);
    y=zeros(columns,2);
    for i=1:columns
        xi=X(:,i);
        probs=findProbs(xi,thetaopt,k);
        [M,I] = max(probs);
        y(i,1)=i;
        y(i,2)=I;   
    end
    prediction=y;   
 end