function H=projectTestdata(W,Xtest,method,lambda)

if nargin<4
    lambda=0;
end

WtW=W'*W;
WtX=W'*Xtest; 
H=zeros(size(W,2),size(Xtest,2));
[noc,I]=size(H);
mu=noc./trace(WtW);
switch method
        case 'SVD'             
            H=WtX;
        case 'AA'                        
            e=ones(noc,1);
            maxiter=50;
            WtW=W'*W;
            WtX=W'*Xtest;
            costold=sum(sum((WtW).*(H*H')))-2*sum(sum(H.*WtX));
            for iter=1:maxiter
               G=WtW*H-WtX;
               G=G-e*sum(G.*H);
               Hold=H;
               mu=noc./trace(WtW);
               stop=0;               
               while ~stop
                  H=Hold-mu*G;
                  H(H<0)=0; 
                  l1_H=sum(H);
                  H=H./(e*l1_H);
                  cost=sum(sum((WtW).*(H*H')))-2*sum(sum(H.*WtX));
                  dcost=costold-cost;
                  if dcost>-1e-12*abs(cost)
                      mu=mu*1.2;
                      costold=cost;          
                      stop=1;
                  else
                      mu=mu/2;
                  end    
               end
            end

        case 'NMF'                                  
            maxiter=50;                        
            costold=sum(sum((WtW).*(H*H')))-2*sum(sum(H.*WtX));            
            for iter=1:maxiter
               G=WtW*H-WtX; 
               Hold=H;
               stop=0;
               while ~stop
                  H=Hold-mu*G;
                  H(H<0)=0;           
                  cost=sum(sum((WtW).*(H*H')))-2*sum(sum(H.*WtX));
                  dcost=costold-cost;            
                  if dcost>=-1e-12*abs(cost)
                      mu=mu*1.2;
                      costold=cost;
                      stop=1;
                  else
                      mu=mu/2;
                  end    
               end
            end           
        case 'SC'
            maxiter=50;
            costold=0.5*(sum(sum((WtW).*(H*H')))-2*sum(sum(H.*WtX)))+lambda*sum(sum(abs(H)));
            for iter=1:maxiter
               G=WtW*H-WtX;    
               Hold=H;
               stop=0;
               while ~stop
                  H=Hold-mu*G;
                  H(abs(H)<mu*lambda)=0;
                  H=H-mu*lambda*sign(H);            
                  L=sum(sum((WtW).*(H*H')))-2*sum(sum(H.*WtX));
                  L1=lambda*sum(sum(abs(H)));
                  cost=0.5*L+L1;
                  dcost=costold-cost;            
                  if dcost>=0
                      mu=mu*1.2;
                      costold=cost;
                      stop=1;
                  else
                      mu=mu/2;
                  end    
               end
            end      
        case 'NSC'
            maxiter=50;
            costold=0.5*(sum(sum((WtW).*(H*H')))-2*sum(sum(H.*WtX)))+lambda*sum(sum(abs(H)));
            for iter=1:maxiter
               G=WtW*H-WtX;    
               Hold=H;
               stop=0;
               while ~stop
                  H=Hold-mu*G;
                  H(abs(H)<mu*lambda)=0;
                  H=H-mu*lambda*sign(H);                             
                  H(H<0)=0;                             
                  L=sum(sum((WtW).*(H*H')))-2*sum(sum(H.*WtX));
                  L1=lambda*sum(sum(abs(H)));
                  cost=0.5*L+L1;
                  dcost=costold-cost;            
                  if dcost>=0
                      mu=mu*1.2;
                      costold=cost;
                      stop=1;
                  else
                      mu=mu/2;
                  end    
               end
            end                 
        case 'kmeans'
            D=sum(W.^2)'*ones(1,size(Xtest,2))-2*WtX+ones(size(W,2),1)*sum(Xtest.^2);
            [val,ind]=min(D);           
            H=full(sparse(ind,1:size(Xtest,2),ones(1,size(Xtest,2))));
        case 'ICA'            
            H=WtW\WtX;
end


