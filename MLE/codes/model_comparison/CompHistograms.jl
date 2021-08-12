using LinearAlgebra
using Polynomials
using Optim
using Distributions
using LsqFit
using Plots
using CSV
using DifferentialEquations
using DelimitedFiles
using StatsPlots
using Measures
using LaTeXStrings
using PyPlot
using GR
using DataFrames
using Dates
using StatsBase
using Pkg
#Pkg.update()



dataFreq_t=readdlm("../data/freq/Data_cleansed/TransnetBW/2018.csv")
dataFreq=dataFreq_t[:,2]
for i in 1:length(dataFreq)
    dataFreq[i]=parse(Float64,dataFreq_t[i,2][10:length(dataFreq_t[i,2])])
end
fs=(dataFreq[:].-50).*2*pi
turnOn=20
posOfdbcrossing=20
solu=Array{Float64,2}(undef,900,2)
sol4=Array{Float64,2}(undef,900,5)
sol5=Array{Float64,2}(undef,900,2)
sol5t=Array{Float64,2}(undef,900,3)


############# read results of all 3 models in #############

dataf3=readdir("./model_atPeak/RES_model_atPeak/")[28:36]
mleres3=Array{Float64,2}(undef,parse(Int64,dataf3[length(dataf3)][15:18]),16)  #8760
for i in 1:length(dataf3)
    le=parse(Int64,dataf3[i][10:13])
    ue=parse(Int64,dataf3[i][15:18])
    mleres3[le:ue,:]=readdlm("./model_atPeak/RES_model_atPeak/"*dataf3[i])[le:ue,:]
end
mleres3


dataf_t=readdir("./model_atDB/RES_model_atDB/")
dataf2=dataf_t[2:length(dataf_t)]
mleres2=Array{Float64,2}(undef,length(dataf2),16)
i=1
for i in 1:length(dataf2)
    mleres2[parse(Int64,dataf2[i][1:length(dataf2[i])-4]),:]=readdlm("./model_atDB/RES_model_atDB/"*dataf2[i])
end
mleres2
Plots.plot(1:8760,mleres2[1:8760,2])


dataf_t1=readdir("./model_always/RES_model_always/")
dataf1=dataf_t1[2:length(dataf_t1)-1]
mleres1=Array{Float64,2}(undef,length(dataf1),16)
i=1
for i in 1:length(dataf1)
    mleres1[parse(Int64,dataf1[i][1:length(dataf1[i])-4]),:]=readdlm("./model_always/RES_model_always/"*dataf1[i])
end
mleres1
Plots.plot(1:4500,mleres1[1:4500,2])



function dyn_moments12!(dy,y,par,t)  #par will look like: [gamma1,gamma2,q,r,D]
    dy[1] = par[3]+(par[4]*t)-par[1]*y[1]-par[2]*y[2]
    dy[2] = y[1]
    dy[3] = par[5]^2-(2*par[1]*y[3])-(2*par[2]*y[5])
    dy[4] = 2*y[5]
    dy[5] = -par[1]*y[5]-par[2]*y[4]+y[3]
end
function dyn_moments1_22!(dy,y,par,t)  #par will look like: [gamma1,gamma2,q,r,D]
    dy[1] = par[3]+(par[4]*t)-par[1]*((t-posOfdbcrossing+1)/turnOn)*y[1]
    dy[2] = y[1]
    dy[3] = par[5]^2-(2*par[1]*((t-posOfdbcrossing+1)/turnOn)*y[3])
    dy[4] = 2*y[5]
    dy[5] = -par[1]*((t-posOfdbcrossing+1)/turnOn)*y[5]+y[3]
end
function dyn_moments02!(dy,y,par,t)  #par will look like: [gamma1,gamma2,q,r,D]
    dy[1] = par[3]+(par[4]*t)
    dy[2] = y[1]
    dy[3] = par[5]^2
    dy[4] = 2*y[5]
    dy[5] = y[3]
end
function timeseries_fpes(posOfdbcrossing,turnOn,par) #with smooth turn on but no theta-values
    if posOfdbcrossing==0       #always in db-->control off
        so=solve(ODEProblem(dyn_moments02!,[par[6],par[7],par[8],par[9],par[10]],(0.0,899.0),par[1:5]))
        for j in 1:900
            global sol4[j,:]=so(j-1.0)
        end
    elseif posOfdbcrossing<0       #always in db-->control off
        so=solve(ODEProblem(dyn_moments12!,[par[6],par[7],par[8],par[9],par[10]],(0.0,899.0),par[1:5]))
        for j in 1:900
            global sol4[j,:]=so(j-1.0)
        end
    else                             # posOfdbcrossing>0 (first inside db, then outside)
        so=solve(ODEProblem(dyn_moments02!,[par[6],par[7],par[8],par[9],par[10]],(0.0,posOfdbcrossing-1),par[1:5]))
        for j in 1:posOfdbcrossing
            global sol4[j,:]=so(j-1.0)
        end
        so=solve(ODEProblem(dyn_moments1_22!,[sol4[posOfdbcrossing,1],sol4[posOfdbcrossing,2],sol4[posOfdbcrossing,3],sol4[posOfdbcrossing,4],sol4[posOfdbcrossing,5]],(posOfdbcrossing-1.0,posOfdbcrossing+(turnOn-1.0)),par[1:5]))
        for j in posOfdbcrossing:(posOfdbcrossing+turnOn)
            global sol4[j,:]=so(j-1.0)
        end
        so=solve(ODEProblem(dyn_moments12!,[sol4[(posOfdbcrossing+turnOn),1],sol4[(posOfdbcrossing+turnOn),2],sol4[(posOfdbcrossing+turnOn),3],sol4[(posOfdbcrossing+turnOn),4],sol4[(posOfdbcrossing+turnOn),5]],(posOfdbcrossing+(turnOn-1.0),899.0),par[1:5]))
        for j in (posOfdbcrossing+turnOn):900
            global sol4[j,:]=so(j-1.0)
        end
    end
    global sol5[:,1]=sol4[:,1]
    global sol5[:,2]=sol4[:,3]
    return sol5
end
function timeseries_fpest(posOfdbcrossing,turnOn,par) #with smooth turn on and theta-values
    if posOfdbcrossing==0       #always in db-->control off
        so=solve(ODEProblem(dyn_moments02!,[par[6],par[7],par[8],par[9],par[10]],(0.0,899.0),par[1:5]))
        for j in 1:900
            global sol4[j,:]=so(j-1.0)
        end
    elseif posOfdbcrossing<0       #always in db-->control off
        so=solve(ODEProblem(dyn_moments12!,[par[6],par[7],par[8],par[9],par[10]],(0.0,899.0),par[1:5]))
        for j in 1:900
            global sol4[j,:]=so(j-1.0)
        end
    else                             # posOfdbcrossing>0 (first inside db, then outside)
        so=solve(ODEProblem(dyn_moments02!,[par[6],par[7],par[8],par[9],par[10]],(0.0,posOfdbcrossing-1),par[1:5]))
        for j in 1:posOfdbcrossing
            global sol4[j,:]=so(j-1.0)
        end
        so=solve(ODEProblem(dyn_moments1_22!,[sol4[posOfdbcrossing,1],sol4[posOfdbcrossing,2],sol4[posOfdbcrossing,3],sol4[posOfdbcrossing,4],sol4[posOfdbcrossing,5]],(posOfdbcrossing-1.0,posOfdbcrossing+(turnOn-1.0)),par[1:5]))
        for j in posOfdbcrossing:(posOfdbcrossing+turnOn)
            global sol4[j,:]=so(j-1.0)
        end
        so=solve(ODEProblem(dyn_moments12!,[sol4[(posOfdbcrossing+turnOn),1],sol4[(posOfdbcrossing+turnOn),2],sol4[(posOfdbcrossing+turnOn),3],sol4[(posOfdbcrossing+turnOn),4],sol4[(posOfdbcrossing+turnOn),5]],(posOfdbcrossing+(turnOn-1.0),899.0),par[1:5]))
        for j in (posOfdbcrossing+turnOn):900
            global sol4[j,:]=so(j-1.0)
        end
    end
    global sol5t[:,1]=sol4[:,1]
    global sol5t[:,2]=sol4[:,3]
    global sol5t[:,3]=sol4[:,2]
    return sol5t
end
function mu_sigma2_t(x,para::AbstractArray)
    solu[:,1]=(para[5]-para[3]/(para[1]-para[2]))*exp.(-x.*para[1]).+para[3]/(para[1]-para[2])*exp.(-x.*para[2])
    solu[:,2]=(para[6]^2-para[4]/(2*para[1]))*exp.(x.*-2*para[1]).+para[4]/(2*para[1])
    return solu
end
function sort_out1(mleres,fs) #gives a list of all intervals where mle worked (no NaNs) and where mle makes sense
    #      1) L small ---> var negative
    j=1
    Lsmall=zeros(length(mleres[:,1]))
    for i in 1:length(mleres[:,1])
        if mleres[i,16]<-1e13 # since all intervaks w/o control have L=-1e-25
            global Lsmall[j]=i
            j+=1
        end
    end
    Lsmall2=Lsmall[1:j-1] # intervals with negative vav
    #      2) comparison of MLE to smoothing model
    #            use AIC Value: 2*k-2ln(L),  with k=number of estimated parameters and ln(L): Log-Likelihood
    #                k_mle= 10 from mle + turn-on-procedure, which makes 1 more??
    #                k_movAv= var
    smoothfs=Array{Float64,1}(undef,900)
    smoothfs2=Array{Float64,1}(undef,900)
    vari=Array{Float64,1}(undef,length(mleres[:,1]))
    likeli=Array{Float64,1}(undef,length(mleres[:,1]))
    likelidiff=Array{Float64,1}(undef,length(mleres[:,1]))
    #AIC_diff=Array{Float64,1}(undef,length(mleres[:,1]))
    for h in 1:length(mleres[:,1])
        for i in 1:70
            global smoothfs[i]=sum(fs[(((h-1)*3600)+1):(((h-1)*3600)+(2*i))])/(2*i)
        end
        for i in 71:830
            global smoothfs[i]=sum(fs[(((h-1)*3600)+(i-70)):(((h-1)*3600)+(i+70))])/140
        end
        for i in 831:900
            global smoothfs[i]=sum(fs[(((h-1)*3600)+900-(2*(900-i))):(((h-1)*3600)+900)])/(2*(900-i)+1)
        end
        for i in 880:900
            global smoothfs[i]=smoothfs[879]
        end
        for i in 1:40
            global smoothfs2[i]=sum(smoothfs[1:(2*i)])/(2*i)
        end
        for i in 41:860
            global smoothfs2[i]=sum(smoothfs[(i-40):(i+40)])/80
        end
        for i in 861:880
            global smoothfs2[i]=sum(smoothfs[(900-(2*(900-i))):900])/(2*(900-i))
        end
        for i in 880:900
            global smoothfs2[i]=smoothfs2[879]
        end
        global vari[h]=var(fs[((h-1)*3600+1):((h-1)*3600+900)].-smoothfs2)
        global likeli[h]= -sum((1/2)*log(2*pi*vari[h]).-(((fs[(3600*(h-1)+1):(3600*(h-1)+900)].-smoothfs2).^2)/(2*vari[h])))
        likelidiff[h]=abs(likeli[h]-mleres[h,16])
        #AIC_diff[h]=abs(2*10-2*mleres[h,16]-(2*1-2*likeli[h]))
        if likelidiff[h]>1e4
            likelidiff[h]=NaN
        end
        #if AIC_diff[h]>2e4
        #    AIC_diff[h]=NaN
        #end
    end
    #plot(1:length(mleres[:,1]),likelidiff[1:length(mleres[:,1])],seriestype = :scatter,ylims=(700,2000))
    #boxplot(filter(!isnan,likelidiff[1:length(mleres[:,1])]),ylims=(700,2000))
    #plot(1:length(mleres[:,1]),AIC_diff[1:length(mleres[:,1])],seriestype = :scatter,ylims=(700,6000))
    #boxplot(filter(!isnan,AIC_diff[1:length(mleres[:,1])]),ylims=(700,6000))
    q10_L=quantile(filter(!isnan,likelidiff[1:length(mleres[:,1])]),0.9)
    #q10_AIC=quantile(filter(!isnan,AIC_diff[1:length(mleres[:,1])]),0.9)
    j=1
    #k=1
    likelidifflarge=Array{Float64,1}(undef,length(mleres[:,1]))
    #AICdifflarge=Array{Float64,1}(undef,length(mleres[:,1]))
    for i in 1:length(mleres[:,1])
        if abs(likelidiff[i])>q10_L || (isnan(likelidiff[i])==true)
            likelidifflarge[j]=i
            j+=1
        end
        #if abs(AIC_diff[i])>q10_AIC
        #    AIC_diff[k]=i
        #    k+=1
        #end
    end
    likelidifflarge2=likelidifflarge[1:j-1]
    #AIC_diff2=AIC_diff[1:j-1]
    #AIC_diff2==likelidifflarge2
    #    bring likelilarge and neg var together:
    goodones2=Array{Float64,1}(undef,length(mleres[:,1]))
    throuwout2=Array{Float64,1}(undef,length(Lsmall2)+length(likelidifflarge2))
    throuwout2[1:length(Lsmall2)]=Lsmall2
    throuwout2[length(Lsmall2)+1:(length(Lsmall2)+length(likelidifflarge2))]=likelidifflarge2
    j=1
    for i in 1:length(mleres[:,1])
        if i in throuwout2
        else
            global goodones2[j]=i
            j+=1
        end
    end
    goodones2
    return goods1=convert.(Int64,goodones2[1:j-1]),likelidiff
end
function sort_out2(mleres,fs) #gives a list of all intervals where mle worked (no NaNs) and where mle makes sense
    #      1) L small ---> var negative
    j=1
    Lsmall=zeros(length(mleres[:,1]))
    for i in 1:length(mleres[:,1])
        if mleres[i,16]<-1e13 # since all intervaks w/o control have L=-1e-25
            global Lsmall[j]=i
            j+=1
        end
    end
    Lsmall2=Lsmall[1:j-1] # intervals with negative vav
    #      2) comparison of MLE to smoothing model
    #            use AIC Value: 2*k-2ln(L),  with k=number of estimated parameters and ln(L): Log-Likelihood
    #                k_mle= 10 from mle + turn-on-procedure, which makes 1 more??
    #                k_movAv= var
    smoothfs=Array{Float64,1}(undef,900)
    smoothfs2=Array{Float64,1}(undef,900)
    vari=Array{Float64,1}(undef,length(mleres[:,1]))
    likeli=Array{Float64,1}(undef,length(mleres[:,1]))
    likelidiff=Array{Float64,1}(undef,length(mleres[:,1]))
    #AIC_diff=Array{Float64,1}(undef,length(mleres[:,1]))
    for h in 1:length(mleres[:,1])
        for i in 1:70
            global smoothfs[i]=sum(fs[(((h-1)*3600)+1):(((h-1)*3600)+(2*i))])/(2*i)
        end
        for i in 71:830
            global smoothfs[i]=sum(fs[(((h-1)*3600)+(i-70)):(((h-1)*3600)+(i+70))])/140
        end
        for i in 831:900
            global smoothfs[i]=sum(fs[(((h-1)*3600)+900-(2*(900-i))):(((h-1)*3600)+900)])/(2*(900-i)+1)
        end
        for i in 880:900
            global smoothfs[i]=smoothfs[879]
        end
        for i in 1:40
            global smoothfs2[i]=sum(smoothfs[1:(2*i)])/(2*i)
        end
        for i in 41:860
            global smoothfs2[i]=sum(smoothfs[(i-40):(i+40)])/80
        end
        for i in 861:880
            global smoothfs2[i]=sum(smoothfs[(900-(2*(900-i))):900])/(2*(900-i))
        end
        for i in 880:900
            global smoothfs2[i]=smoothfs2[879]
        end
        global vari[h]=var(fs[((h-1)*3600+1):((h-1)*3600+900)].-smoothfs2)
        global likeli[h]= -sum((1/2)*log(2*pi*vari[h]).-(((fs[(3600*(h-1)+1):(3600*(h-1)+900)].-smoothfs2).^2)/(2*vari[h])))
        likelidiff[h]=abs(likeli[h]-mleres[h,16])
        #AIC_diff[h]=abs(2*10-2*mleres[h,16]-(2*1-2*likeli[h]))
        if likelidiff[h]>1e4
            likelidiff[h]=NaN
        end
        #if AIC_diff[h]>2e4
        #    AIC_diff[h]=NaN
        #end
    end
    #plot(1:length(mleres[:,1]),likelidiff[1:length(mleres[:,1])],seriestype = :scatter,ylims=(700,2000))
    #boxplot(filter(!isnan,likelidiff[1:length(mleres[:,1])]),ylims=(700,2000))
    #plot(1:length(mleres[:,1]),AIC_diff[1:length(mleres[:,1])],seriestype = :scatter,ylims=(700,6000))
    #boxplot(filter(!isnan,AIC_diff[1:length(mleres[:,1])]),ylims=(700,6000))
    q10_L=quantile(filter(!isnan,likelidiff[1:length(mleres[:,1])]),0.9)
    #q10_AIC=quantile(filter(!isnan,AIC_diff[1:length(mleres[:,1])]),0.9)
    j=1
    #k=1
    likelidifflarge=Array{Float64,1}(undef,length(mleres[:,1]))
    #AICdifflarge=Array{Float64,1}(undef,length(mleres[:,1]))
    for i in 1:length(mleres[:,1])
        if abs(likelidiff[i])>q10_L || (isnan(likelidiff[i])==true)
            likelidifflarge[j]=i
            j+=1
        end
        #if abs(AIC_diff[i])>q10_AIC
        #    AIC_diff[k]=i
        #    k+=1
        #end
    end
    likelidifflarge2=likelidifflarge[1:j-1]
    #AIC_diff2=AIC_diff[1:j-1]
    #AIC_diff2==likelidifflarge2
    #    bring likelilarge and neg var together:
    goodones2=Array{Float64,1}(undef,length(mleres[:,1]))
    throuwout2=Array{Float64,1}(undef,length(Lsmall2)+length(likelidifflarge2))
    throuwout2[1:length(Lsmall2)]=Lsmall2
    throuwout2[length(Lsmall2)+1:(length(Lsmall2)+length(likelidifflarge2))]=likelidifflarge2
    j=1
    for i in 1:length(mleres[:,1])
        if i in throuwout2
        else
            global goodones2[j]=i
            j+=1
        end
    end
    goodones2
    return goods2=convert.(Int64,goodones2[1:j-1]),likelidiff
end
function sort_out3(mleres,fs) #gives a list of all intervals where mle worked (no NaNs) and where mle makes sense
    #      1) L small ---> var negative
    j=1
    Lsmall=zeros(length(mleres[:,1]))
    for i in 1:length(mleres[:,1])
        if mleres[i,16]<-1e13 # since all intervaks w/o control have L=-1e-25
            global Lsmall[j]=i
            j+=1
        end
    end
    Lsmall2=Lsmall[1:j-1] # intervals with negative vav
    #      2) comparison of MLE to smoothing model
    #            use AIC Value: 2*k-2ln(L),  with k=number of estimated parameters and ln(L): Log-Likelihood
    #                k_mle= 10 from mle + turn-on-procedure, which makes 1 more??
    #                k_movAv= var
    smoothfs=Array{Float64,1}(undef,900)
    smoothfs2=Array{Float64,1}(undef,900)
    vari=Array{Float64,1}(undef,length(mleres[:,1]))
    likeli=Array{Float64,1}(undef,length(mleres[:,1]))
    likelidiff=Array{Float64,1}(undef,length(mleres[:,1]))
    #AIC_diff=Array{Float64,1}(undef,length(mleres[:,1]))
    for h in 1:length(mleres[:,1])
        for i in 1:70
            global smoothfs[i]=sum(fs[(((h-1)*3600)+1):(((h-1)*3600)+(2*i))])/(2*i)
        end
        for i in 71:830
            global smoothfs[i]=sum(fs[(((h-1)*3600)+(i-70)):(((h-1)*3600)+(i+70))])/140
        end
        for i in 831:900
            global smoothfs[i]=sum(fs[(((h-1)*3600)+900-(2*(900-i))):(((h-1)*3600)+900)])/(2*(900-i)+1)
        end
        for i in 880:900
            global smoothfs[i]=smoothfs[879]
        end
        for i in 1:40
            global smoothfs2[i]=sum(smoothfs[1:(2*i)])/(2*i)
        end
        for i in 41:860
            global smoothfs2[i]=sum(smoothfs[(i-40):(i+40)])/80
        end
        for i in 861:880
            global smoothfs2[i]=sum(smoothfs[(900-(2*(900-i))):900])/(2*(900-i))
        end
        for i in 880:900
            global smoothfs2[i]=smoothfs2[879]
        end
        global vari[h]=var(fs[((h-1)*3600+1):((h-1)*3600+900)].-smoothfs2)
        global likeli[h]= -sum((1/2)*log(2*pi*vari[h]).-(((fs[(3600*(h-1)+1):(3600*(h-1)+900)].-smoothfs2).^2)/(2*vari[h])))
        likelidiff[h]=abs(likeli[h]-mleres[h,16])
        #AIC_diff[h]=abs(2*10-2*mleres[h,16]-(2*1-2*likeli[h]))
        if likelidiff[h]>1e4
            likelidiff[h]=NaN
        end
        #if AIC_diff[h]>2e4
        #    AIC_diff[h]=NaN
        #end
    end
    #plot(1:length(mleres[:,1]),likelidiff[1:length(mleres[:,1])],seriestype = :scatter,ylims=(700,2000))
    #boxplot(filter(!isnan,likelidiff[1:length(mleres[:,1])]),ylims=(700,2000))
    #plot(1:length(mleres[:,1]),AIC_diff[1:length(mleres[:,1])],seriestype = :scatter,ylims=(700,6000))
    #boxplot(filter(!isnan,AIC_diff[1:length(mleres[:,1])]),ylims=(700,6000))
    q10_L=quantile(filter(!isnan,likelidiff[1:length(mleres[:,1])]),0.9)
    #q10_AIC=quantile(filter(!isnan,AIC_diff[1:length(mleres[:,1])]),0.9)
    j=1
    #k=1
    likelidifflarge=Array{Float64,1}(undef,length(mleres[:,1]))
    #AICdifflarge=Array{Float64,1}(undef,length(mleres[:,1]))
    for i in 1:length(mleres[:,1])
        if abs(likelidiff[i])>q10_L || (isnan(likelidiff[i])==true)
            likelidifflarge[j]=i
            j+=1
        end
        #if abs(AIC_diff[i])>q10_AIC
        #    AIC_diff[k]=i
        #    k+=1
        #end
    end
    likelidifflarge2=likelidifflarge[1:j-1]
    #AIC_diff2=AIC_diff[1:j-1]
    #AIC_diff2==likelidifflarge2
    #    bring likelilarge and neg var together:
    goodones2=Array{Float64,1}(undef,length(mleres[:,1]))
    throuwout2=Array{Float64,1}(undef,length(Lsmall2)+length(likelidifflarge2))
    throuwout2[1:length(Lsmall2)]=Lsmall2
    throuwout2[length(Lsmall2)+1:(length(Lsmall2)+length(likelidifflarge2))]=likelidifflarge2
    j=1
    for i in 1:length(mleres[:,1])
        if i in throuwout2
        else
            global goodones2[j]=i
            j+=1
        end
    end
    goodones2
    return goods3=convert.(Int64,goodones2[1:j-1]),likelidiff
end
function density_plot(xli,yli,xdat,ydat,zdat,gridx,gridy)#(xl,xu)=xlims(plo)
    xd=(xli[2]-xli[1])/gridx
    yd=(yli[2]-yli[1])/gridy
    xval=zeros(length(xdat))
    yval=zeros(length(xdat))
    zval=zeros(length(xdat))
    c=1
    for i in 1:gridx
        for j in 1:gridy
            counte=0
            ztemp=0
            for k in 1:length(xdat)
                if (xli[1]+xd*(i-1))<xdat[k]<(xli[1]+xd*i)
                    if (yli[1]+yd*(j-1))<ydat[k]<(yli[1]+yd*j)
                        ztemp+=zdat[k]
                        counte+=1
                    end
                end
            end
            ztemp=ztemp/counte
            if isnan(ztemp)==false
                xval[c]=(xli[1]+xd*(i-0.5))
                yval[c]=(yli[1]+yd*(j-0.5))
                zval[c]=ztemp
                c+=1
            end
        end
    end
    return (xval[1:c-1],yval[1:c-1],zval[1:c-1])
end
function plotInt1(hour)
    global turnOn=convert(Int64,mleres1[hour,1])
    global posOfdbcrossing=convert(Int64,mleres1[hour,2])
    mleres1[hour,8]
    TSdp=Array{Float64,1}(undef,900)
    TSdpt=Array{Float64,1}(undef,900)
    #TSdp[1:100].=NaN
    #TSdpt[1:100].=NaN
    TS=timeseries_fpest(convert(Int64,mleres1[hour,2]),convert(Int64,mleres1[hour,1]),mleres1[hour,3:12])
    for i in 1:900
         TSdp[i]=mleres1[hour,5]+mleres1[hour,6]*(i-1)
         if i<posOfdbcrossing+turnOn
             TSdpt[i]=TSdp[i]
         else
             TSdpt[i]=TSdp[i]-mleres1[hour,4]*TS[i,3]
         end
    end
    TSsigma = sqrt.(abs.(TS[:,2]))
    TSmuPlus = TS[:,1]+TSsigma[:]
    TSmuMinus = TS[:,1]-TSsigma[:]
    yupper=maximum([maximum(TSdp[1:900]),maximum(TSdpt[1:900]),0])
    ylower=minimum([minimum(TSdp[1:900]),minimum(TSdpt[1:900]),0])
    inform="g1= "*string(round(mleres1[hour,3];digits=5))*"   g2= "*string(round(mleres1[hour,4];digits=5))*"   D= "*string(round(mleres1[hour,7];digits=3))*"   r= "*string(round(mleres1[hour,6];digits=7))*"   L= "*string(round(mleres1[hour,16];digits=0))*"   AIC= "*string(round((2*10-2*mleres1[hour,16]);digits=0))
    ueberschrift="\$\\textrm{model \\ 1, \\ hour \\ }"*string(hour)*"\$"
    #path=".\plots\first70\combi"*string(hour)
    p1=Plots.plot(0:899,[TS[:,1] TSmuPlus TSmuMinus],color=[:blue :lightblue :lightblue],style=[:solid :dash :dash],label=false,ylabel=L"\omega",xaxis=false,title=ueberschrift)
    Plots.plot!(0:899,fs[((hour-1)*3600+1):((hour-1)*3600+900)],label=false,color=:black)
    p2=Plots.plot(0:899,[TSdp TSdpt],color=[:blue :lightblue],label=[L"\Delta \textrm{P}" L"\Delta \textrm{P}+2\textrm{nd}"],ylims=(ylower,yupper),xlabel=L"\textrm{time \ in \ s}") #,annotations = (0,yupper,Plots.text(inform, :left,10))
    Plots.plot(p1,p2,layout = (2, 1))
end
function plotInt2(hour)
    global turnOn=convert(Int64,mleres2[hour,1])
    global posOfdbcrossing=convert(Int64,mleres2[hour,2])
    mleres2[hour,8]
    TSdp=Array{Float64,1}(undef,900)
    TSdpt=Array{Float64,1}(undef,900)
    #TSdp[1:100].=NaN
    #TSdpt[1:100].=NaN
    TS=timeseries_fpest(convert(Int64,mleres2[hour,2]),convert(Int64,mleres2[hour,1]),mleres2[hour,3:12])
    for i in 1:900
         TSdp[i]=mleres2[hour,5]+mleres2[hour,6]*(i-1)
         if i<posOfdbcrossing+turnOn
             TSdpt[i]=TSdp[i]
         else
             TSdpt[i]=TSdp[i]-mleres2[hour,4]*TS[i,3]
         end
    end
    TSsigma = sqrt.(abs.(TS[:,2]))
    TSmuPlus = TS[:,1]+TSsigma[:]
    TSmuMinus = TS[:,1]-TSsigma[:]
    yupper=maximum([maximum(TSdp[1:900]),maximum(TSdpt[1:900]),0])
    ylower=minimum([minimum(TSdp[1:900]),minimum(TSdpt[1:900]),0])
    inform="g1= "*string(round(mleres2[hour,3];digits=5))*"   g2= "*string(round(mleres2[hour,4];digits=5))*"   D= "*string(round(mleres2[hour,7];digits=3))*"   r= "*string(round(mleres2[hour,6];digits=7))*"   L= "*string(round(mleres2[hour,16];digits=0))*"   AIC= "*string(round((2*10-2*mleres2[hour,16]);digits=0))
    ueberschrift="\$\\textrm{model \\ 2, \\ hour \\ }"*string(hour)*"\$"
    #path=".\plots\first70\combi"*string(hour)
    p1=Plots.plot(0:899,[TS[:,1] TSmuPlus TSmuMinus],color=[:blue :lightblue :lightblue],style=[:solid :dash :dash],label=false,ylabel=L"\omega",xaxis=false,title=ueberschrift)
    Plots.plot!(0:899,fs[((hour-1)*3600+1):((hour-1)*3600+900)],label=false,color=:black)
    p2=Plots.plot(0:899,[TSdp TSdpt],color=[:blue :lightblue],label=[L"\Delta \textrm{P}" L"\Delta \textrm{P}+2\textrm{nd}"],ylims=(ylower,yupper),xlabel=L"\textrm{time \ in \ s}") #,annotations = (0,yupper,Plots.text(inform, :left,10))
    Plots.plot(p1,p2,layout = (2, 1))
end
function plotInt3(hour)
    global turnOn=convert(Int64,mleres3[hour,1])
    global posOfdbcrossing=convert(Int64,mleres3[hour,2])
    mleres3[hour,8]
    TSdp=Array{Float64,1}(undef,1000)
    TSdpt=Array{Float64,1}(undef,1000)
    TSdp[1:100].=NaN
    TSdpt[1:100].=NaN
    TS=timeseries_fpest(convert(Int64,mleres3[hour,2]),convert(Int64,mleres3[hour,1]),mleres3[hour,3:12])
    for i in 1:900
         TSdp[i+100]=mleres3[hour,5]+mleres3[hour,6]*(i-1)
         if i<posOfdbcrossing+turnOn
             TSdpt[i+100]=TSdp[i+100]
         else
             TSdpt[i+100]=TSdp[i+100]-mleres3[hour,4]*TS[i,3]
         end
    end
    TSsigma = sqrt.(abs.(TS[:,2]))
    TSmuPlus = TS[:,1]+TSsigma[:]
    TSmuMinus = TS[:,1]-TSsigma[:]
    yupper=maximum([maximum(TSdp[101:1000]),maximum(TSdpt[101:1000]),0])
    ylower=minimum([minimum(TSdp[101:1000]),minimum(TSdpt[101:1000]),0])
    inform="g1= "*string(round(mleres3[hour,3];digits=5))*"   g2= "*string(round(mleres3[hour,4];digits=5))*"   D= "*string(round(mleres3[hour,7];digits=3))*"   r= "*string(round(mleres3[hour,6];digits=7))*"   L= "*string(round(mleres3[hour,16];digits=0))*"   AIC= "*string(round((2*10-2*mleres3[hour,16]);digits=0))
    ueberschrift="\$\\textrm{model \\ 3, \\ hour \\ }"*string(hour)*"\$"
    #path=".\plots\first70\combi"*string(hour)
    p1=Plots.plot(0:899,[TS[:,1] TSmuPlus TSmuMinus],color=[:blue :lightblue :lightblue],style=[:solid :dash :dash],label=false,ylabel=L"\omega",xaxis=false,title=ueberschrift)
    Plots.plot!(0:899,fs[((hour-1)*3600+1):((hour-1)*3600+900)],label=false,color=:black)
    p2=Plots.plot(-100:899,[TSdp TSdpt],color=[:blue :lightblue],label=[L"\Delta \textrm{P}" L"\Delta \textrm{P}+2\textrm{nd}"],ylims=(ylower,yupper),xlabel=L"\textrm{time \ in \ s}") #,annotations = (0,yupper,Plots.text(inform, :left,10))
    Plots.plot(p1,p2,layout = (2, 1))
end

function plotInt1w(hour)
    global turnOn=convert(Int64,mleres1[hour,1])
    global posOfdbcrossing=convert(Int64,mleres1[hour,2])
    mleres1[hour,8]
    TSdp=Array{Float64,1}(undef,900)
    TSdpt=Array{Float64,1}(undef,900)
    #TSdp[1:100].=NaN
    #TSdpt[1:100].=NaN
    TS=timeseries_fpest(convert(Int64,mleres1[hour,2]),convert(Int64,mleres1[hour,1]),mleres1[hour,3:12])
    for i in 1:900
         TSdp[i]=mleres1[hour,5]+mleres1[hour,6]*(i-1)
         if i<posOfdbcrossing+turnOn
             TSdpt[i]=TSdp[i]
         else
             TSdpt[i]=TSdp[i]-mleres1[hour,4]*TS[i,3]
         end
    end
    TSsigma = sqrt.(abs.(TS[:,2]))
    TSmuPlus = TS[:,1]+TSsigma[:]
    TSmuMinus = TS[:,1]-TSsigma[:]
    yupper=maximum([maximum(TSdp[1:900]),maximum(TSdpt[1:900]),0])
    ylower=minimum([minimum(TSdp[1:900]),minimum(TSdpt[1:900]),0])
    inform="g1= "*string(round(mleres1[hour,3];digits=5))*"   g2= "*string(round(mleres1[hour,4];digits=5))*"   D= "*string(round(mleres1[hour,7];digits=3))*"   r= "*string(round(mleres1[hour,6];digits=7))*"   L= "*string(round(mleres1[hour,16];digits=0))*"   AIC= "*string(round((2*10-2*mleres1[hour,16]);digits=0))
    ueberschrift="\$\\textrm{model \\ 1, \\ hour \\ }"*string(hour)*"\$"
    #path=".\plots\first70\combi"*string(hour)
    p1=Plots.plot(0:899,[TS[:,1] TSmuPlus TSmuMinus],color=[:blue :lightblue :lightblue],style=[:solid :dash :dash],label=false,xlabel=L"\textrm{time \ in \ s}",ylabel=L"\omega",title=ueberschrift)
    Plots.plot!(0:899,fs[((hour-1)*3600+1):((hour-1)*3600+900)],label=false,color=:black)
    #p2=Plots.plot(0:899,[TSdp TSdpt],color=[:blue :lightblue],label=[L"\Delta \textrm{P}" L"\Delta \textrm{P}+2\textrm{nd}"],ylims=(ylower,yupper),xlabel=L"\textrm{time \ in \ s}") #,annotations = (0,yupper,Plots.text(inform, :left,10))
    #Plots.plot(p1,p2,layout = (2, 1))
end
function plotInt2w(hour)
    global turnOn=convert(Int64,mleres2[hour,1])
    global posOfdbcrossing=convert(Int64,mleres2[hour,2])
    mleres2[hour,8]
    TSdp=Array{Float64,1}(undef,900)
    TSdpt=Array{Float64,1}(undef,900)
    #TSdp[1:100].=NaN
    #TSdpt[1:100].=NaN
    TS=timeseries_fpest(convert(Int64,mleres2[hour,2]),convert(Int64,mleres2[hour,1]),mleres2[hour,3:12])
    for i in 1:900
         TSdp[i]=mleres2[hour,5]+mleres2[hour,6]*(i-1)
         if i<posOfdbcrossing+turnOn
             TSdpt[i]=TSdp[i]
         else
             TSdpt[i]=TSdp[i]-mleres2[hour,4]*TS[i,3]
         end
    end
    TSsigma = sqrt.(abs.(TS[:,2]))
    TSmuPlus = TS[:,1]+TSsigma[:]
    TSmuMinus = TS[:,1]-TSsigma[:]
    yupper=maximum([maximum(TSdp[1:900]),maximum(TSdpt[1:900]),0])
    ylower=minimum([minimum(TSdp[1:900]),minimum(TSdpt[1:900]),0])
    inform="g1= "*string(round(mleres2[hour,3];digits=5))*"   g2= "*string(round(mleres2[hour,4];digits=5))*"   D= "*string(round(mleres2[hour,7];digits=3))*"   r= "*string(round(mleres2[hour,6];digits=7))*"   L= "*string(round(mleres2[hour,16];digits=0))*"   AIC= "*string(round((2*10-2*mleres2[hour,16]);digits=0))
    ueberschrift="\$\\textrm{model \\ 2, \\ hour \\ }"*string(hour)*"\$"
    #path=".\plots\first70\combi"*string(hour)
    p1=Plots.plot(0:899,[TS[:,1] TSmuPlus TSmuMinus],color=[:blue :lightblue :lightblue],style=[:solid :dash :dash],label=false,ylabel=L"\omega",xlabel=L"\textrm{time \ in \ s}",title=ueberschrift)
    Plots.plot!(0:899,fs[((hour-1)*3600+1):((hour-1)*3600+900)],label=false,color=:black)
    #p2=Plots.plot(0:899,[TSdp TSdpt],color=[:blue :lightblue],label=[L"\Delta \textrm{P}" L"\Delta \textrm{P}+2\textrm{nd}"],ylims=(ylower,yupper),xlabel=L"\textrm{time \ in \ s}") #,annotations = (0,yupper,Plots.text(inform, :left,10))
    #Plots.plot(p1,p2,layout = (2, 1))
end
function plotInt3w(hour)
    global turnOn=convert(Int64,mleres3[hour,1])
    global posOfdbcrossing=convert(Int64,mleres3[hour,2])
    mleres3[hour,8]
    TSdp=Array{Float64,1}(undef,1000)
    TSdpt=Array{Float64,1}(undef,1000)
    TSdp[1:100].=NaN
    TSdpt[1:100].=NaN
    TS=timeseries_fpest(convert(Int64,mleres3[hour,2]),convert(Int64,mleres3[hour,1]),mleres3[hour,3:12])
    for i in 1:900
         TSdp[i+100]=mleres3[hour,5]+mleres3[hour,6]*(i-1)
         if i<posOfdbcrossing+turnOn
             TSdpt[i+100]=TSdp[i+100]
         else
             TSdpt[i+100]=TSdp[i+100]-mleres3[hour,4]*TS[i,3]
         end
    end
    TSsigma = sqrt.(abs.(TS[:,2]))
    TSmuPlus = TS[:,1]+TSsigma[:]
    TSmuMinus = TS[:,1]-TSsigma[:]
    yupper=maximum([maximum(TSdp[101:1000]),maximum(TSdpt[101:1000]),0])
    ylower=minimum([minimum(TSdp[101:1000]),minimum(TSdpt[101:1000]),0])
    inform="g1= "*string(round(mleres3[hour,3];digits=5))*"   g2= "*string(round(mleres3[hour,4];digits=5))*"   D= "*string(round(mleres3[hour,7];digits=3))*"   r= "*string(round(mleres3[hour,6];digits=7))*"   L= "*string(round(mleres3[hour,16];digits=0))*"   AIC= "*string(round((2*10-2*mleres3[hour,16]);digits=0))
    ueberschrift="\$\\textrm{model \\ 3, \\ hour \\ }"*string(hour)*"\$"
    #path=".\plots\first70\combi"*string(hour)
    p1=Plots.plot(0:899,[TS[:,1] TSmuPlus TSmuMinus],color=[:blue :lightblue :lightblue],style=[:solid :dash :dash],label=false,ylabel=L"\omega",xlabel=L"\textrm{time \ in \ s}",title=ueberschrift)
    Plots.plot!(0:899,fs[((hour-1)*3600+1):((hour-1)*3600+900)],label=false,color=:black)
    #p2=Plots.plot(-100:899,[TSdp TSdpt],color=[:blue :lightblue],label=[L"\Delta \textrm{P}" L"\Delta \textrm{P}+2\textrm{nd}"],ylims=(ylower,yupper),xlabel=L"\textrm{time \ in \ s}") #,annotations = (0,yupper,Plots.text(inform, :left,10))
    #Plots.plot(p1,p2,layout = (2, 1))
end
################## sort out the bad fits ##################

(goods1,likelidiff)=sort_out1(mleres1,fs)
Plots.plot(goods1,likelidiff[goods1],seriestype=:scatter)
goods1

(goods2,likelidiff)=sort_out2(mleres2,fs)
Plots.plot(goods2,likelidiff[goods2],seriestype=:scatter)
goods2

(goods3,likelidiff)=sort_out3(mleres3,fs)
Plots.plot(goods3,likelidiff[goods3],seriestype=:scatter)
goods3





################## make the the histograms ################

goods1
goods2
goods3
goods12=intersect(goods1,goods2)
goods23=intersect(goods2,goods3)
goods13=intersect(goods1,goods3)


likelidiff12=mleres1[goods12,16]-mleres2[goods12,16]
Plots.histogram(-likelidiff12,title="\$\\textrm{Difference \\ of \\ the \\ fit \\ quality}\$",label=false,xlabel="\$\\textrm{AIC}_1-\\textrm{AIC}_2\$")
png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_comparison/plots/hist12.png")
mean(-likelidiff12)

likelidiff23=mleres2[goods23,16]-mleres3[goods23,16]
Plots.histogram(-likelidiff23,title="\$\\textrm{Difference \\ of \\ the \\ fit \\ quality}\$",label=false,xlabel="\$\\textrm{AIC}_2-\\textrm{AIC}_3\$")
png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_comparison/plots/hist23.png")
mean(-likelidiff23)

likelidiff13=mleres1[goods13,16]-mleres3[goods13,16]
Plots.histogram(-likelidiff13,title="\$\\textrm{Difference \\ of \\ the \\ fit \\ quality}\$",label=false,xlabel="\$\\textrm{AIC}_1-\\textrm{AIC}_3\$")
png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_comparison/plots/hist13.png")
mean(-likelidiff13)


################ check, why model 1 and 2 give nearly always the same result ###############
mleres1[:,2]
whereStart_t=zeros(length(mleres2[:,2]))
j=1
for i in 1:8760
    if mleres2[i,2] > -1
        whereStart_t[j]=convert(Int64,i)
        j+=1
    end
end
controlOn=convert.(Int64,whereStart_t[1:j-1])
goods12Con=intersect(goods1,goods2,controlOn)

i=21
plotInt1(goods12Con[i])
plotInt2(goods12Con[i])
mleres2[i,:]

likelidiff12Con=mleres1[goods12Con,16]-mleres2[goods12Con,16]
Plots.histogram(-likelidiff12Con,title="\$\\textrm{Difference \\ of \\ fit \\ quality}\$",label=false,xlabel="\$\\textrm{AIC}_1-\\textrm{AIC}_2\$")
png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_comparison/plots/hist12Con.png")
mean(-likelidiff12Con)


########### comparison of 2,3 #############

better2=Array{Int64,1}(undef,length(filter!(x -> x > 300, likelidiff23)))
likelidiff23=mleres2[goods23,16]-mleres3[goods23,16]
j=1
for i in 1:length(goods23)
    if likelidiff23[i]>300
        better2[j]=goods23[i]
        j+=1
    end
end
better2

better3=Array{Int64,1}(undef,length(filter!(x -> x < -800, likelidiff23)))
likelidiff23=mleres2[goods23,16]-mleres3[goods23,16]
j=1
for i in 1:length(goods23)
    if likelidiff23[i]<-800
        better3[j]=goods23[i]
        j+=1
    end
end
better3

i=17
plotInt2(better2[i])
plotInt3(better2[i])
beautiful: better2[2,3,4,5,6,7,10,13]
i=20
plotInt2(better3[i])
plotInt3(better3[i])
beautiful: better3[2,4,5,13,17,20   ,109]


i=17
plotInt2w(better2[i])
plotInt3w(better2[i])
beautiful: better2[2,3,4,5,6,7,10,13]
i=17
plotInt2w(better3[i])
plotInt3w(better3[i])
beautiful: better3[2,4,5,13,17,20   ,109]
png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_comparison/plots/23_m3better2.png")
png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_comparison/plots/23_m2better2.png")

plotInt1w(226)
plotInt3w(226)
mleres3[2]
########### comparison of 1,2 #############

better1=Array{Int64,1}(undef,length(filter!(x -> x > 300, likelidiff12)))
likelidiff12=mleres1[goods12,16]-mleres2[goods12,16]
j=1
for i in 1:length(goods12)
    if likelidiff12[i]>300
        better1[j]=goods12[i]
        j+=1
    end
end
better2

better3=Array{Int64,1}(undef,length(filter!(x -> x < -800, likelidiff23)))
likelidiff23=mleres2[goods23,16]-mleres3[goods23,16]
j=1
for i in 1:length(goods23)
    if likelidiff23[i]<-800
        better3[j]=goods23[i]
        j+=1
    end
end
better3

i=17
plotInt2(better2[i])
plotInt3(better2[i])
beautiful: better2[2,3,4,5,6,7,10,13]
i=17
plotInt2(better3[i])
plotInt3(better3[i])
beautiful: better3[2,4,5,13,17,20   ,109]
