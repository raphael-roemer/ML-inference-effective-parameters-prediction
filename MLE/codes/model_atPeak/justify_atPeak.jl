Plots.plot(mleres[:,5],mleres[:,6],seriestype=:scatter)
Plots.plot(mleres[goods2,5],mleres[goods2,6],markerstrokewidth = 0,alpha=0.2,seriestype=:scatter)


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
using HDF5
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
mleres=Array{Float64,2}(undef,8760,16)

dataf=readdir("./model_atPeak/RES_model_atPeak/")[28:36]
mleres=Array{Float64,2}(undef,parse(Int64,dataf[length(dataf)][15:18]),16)  #8760
for i in 1:length(dataf)
    le=parse(Int64,dataf[i][10:13])
    ue=parse(Int64,dataf[i][15:18])
    mleres[le:ue,:]=readdlm("./model_atPeak/RES_model_atPeak/"*dataf[i])[le:ue,:]
end
#mleres=mleres[1:200,:]
for i in 1:length(mleres[:,2])
    if isnan(mleres[i,2])==true
        print(i)
        print(" LS ")
    end
end
print("bla")
writedlm("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_always/peakpos.csv",mleres[:,2])

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
function timeseries_fpe(posOfdbcrossing,par)
    if posOfdbcrossing==-1          #always outside db-->control on
        so=solve(ODEProblem(dyn_moments12!,[par[6],par[7],par[8],par[9],par[10]],(0.0,899.0),par[1:5]))
        for j in 1:900
            global sol4[j,:]=so(j-1.0)
        end
    elseif posOfdbcrossing==0       #always in db-->control off
        so=solve(ODEProblem(dyn_moments02!,[par[6],par[7],par[8],par[9],par[10]],(0.0,899.0),par[1:5]))
        for j in 1:900
            global sol4[j,:]=so(j-1.0)
        end
    else                             # posOfdbcrossing>0 (first inside db, then outside)
        so=solve(ODEProblem(dyn_moments02!,[par[6],par[7],par[8],par[9],par[10]],(0.0,posOfdbcrossing-1),par[1:5]))
        for j in 1:posOfdbcrossing
            global sol4[j,:]=so(j-1.0)
        end
        so=solve(ODEProblem(dyn_moments12!,[sol4[posOfdbcrossing,1],sol4[posOfdbcrossing,2],sol4[posOfdbcrossing,3],sol4[posOfdbcrossing,4],sol4[posOfdbcrossing,5]],(posOfdbcrossing-1,899.0),par[1:5]))
        for j in posOfdbcrossing:900
            global sol4[j,:]=so(j-1.0)
        end
    end
    global sol5[:,1]=sol4[:,1]
    global sol5[:,2]=sol4[:,3]
    return sol5 #not smooth turn on
end
function timeseries_fpet(posOfdbcrossing,par)
    if posOfdbcrossing==-1          #always outside db-->control on
        so=solve(ODEProblem(dyn_moments12!,[par[6],par[7],par[8],par[9],par[10]],(0.0,899.0),par[1:5]))
        for j in 1:900
            global sol4[j,:]=so(j-1.0)
        end
    elseif posOfdbcrossing==0       #always in db-->control off
        so=solve(ODEProblem(dyn_moments02!,[par[6],par[7],par[8],par[9],par[10]],(0.0,899.0),par[1:5]))
        for j in 1:900
            global sol4[j,:]=so(j-1.0)
        end
    else                             # posOfdbcrossing>0 (first inside db, then outside)
        so=solve(ODEProblem(dyn_moments02!,[par[6],par[7],par[8],par[9],par[10]],(0.0,posOfdbcrossing-1),par[1:5]))
        for j in 1:posOfdbcrossing
            global sol4[j,:]=so(j-1.0)
        end
        so=solve(ODEProblem(dyn_moments12!,[sol4[posOfdbcrossing,1],sol4[posOfdbcrossing,2],sol4[posOfdbcrossing,3],sol4[posOfdbcrossing,4],sol4[posOfdbcrossing,5]],(posOfdbcrossing-1,899.0),par[1:5]))
        for j in posOfdbcrossing:900
            global sol4[j,:]=so(j-1.0)
        end
    end
    global sol5t[:,1]=sol4[:,1]
    global sol5t[:,2]=sol4[:,3]
    global sol5t[:,3]=sol4[:,2]
    return sol5t #not smooth turn on
end
function mu_sigma2_t(x,para::AbstractArray)
    solu[:,1]=(para[5]-para[3]/(para[1]-para[2]))*exp.(-x.*para[1]).+para[3]/(para[1]-para[2])*exp.(-x.*para[2])
    solu[:,2]=(para[6]^2-para[4]/(2*para[1]))*exp.(x.*-2*para[1]).+para[4]/(2*para[1])
    return solu
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

#= compare different models
###################################################################################
#    x-axis: nadir   y-axis: loglike_combi / loglike_full (just if both L are not 0)
####################################################################################
plot(info_combi[1:70,13],(info_combi[1:70,15].-info_full[1:70,15]),seriestype = :scatter, xlabel="nadir", ylabel="L_com/L_full",ylims=(-50,50.0))

####################################################################################
#    x-axis: nadir   y-axis: AIC_combi / AIC_noPrim (just if both L are not 0)
####################################################################################
plot(info_combi[1:70,13],(((info_combi[1:70,15].*2).+(2*9)).-((info_noPrim[1:70,15].*2).+2*8)),seriestype = :scatter, ylims=(-1000,4000.0),xlabel="nadir", ylabel="L_com/L_noPrim")

####################################################################################
#    x-axis: nadir   y-axis: AIC_full / AIC_noPrim (just if both L are not 0)
####################################################################################
plot(info_combi[1:70,13],(((info_full[1:70,15].*2).+(2*9))./((info_noPrim[1:70,14].*2).+2*8)),seriestype = :scatter,xlims=(0.0,0.3),ylims=(-10.0,3.0))

####################################################################################
#    x-axis: nadir   y-axis: AIC_combi / AIC_effective (just if both L are not 0)
####################################################################################
plot(info_combi[1:70,13],(((info_combi[1:70,15].*2).+(2*6))./((info_effective[1:70,10].*2).+2*9)),seriestype = :scatter, xlims=(0.0,0.9),ylims=(-5,3.0),xlabel="nadir", ylabel="L_com/L_eff")

####################################################################################
#    x-axis: nadir   y-axis: AIC_full / AIC_effective (just if both L are not 0)
####################################################################################
plot(info_combi[1:70,13],(((info_full[1:70,14].*2).+(2*6))./((info_effective[1:70,10].*2).+2*9)),seriestype = :scatter)





####################################################################################
#    x-axis: nadir   y-axis: loglike_combi
####################################################################################
plot(info_combi[1:70,13],info_combi[1:70,15],seriestype = :scatter,xlabel="nadir", ylabel="L_combi",xlims=(0.0,0.9),ylims=(-5,3000.0))
plot(info_combi[1:70,13],info_full[1:70,15],seriestype = :scatter,xlabel="nadir", ylabel="L_full",xlims=(0.0,0.9),ylims=(-5,3000.0))



####################################################################################
#    x-axis: interval-nr   y-axis: D
####################################################################################
plot(1:70,combi70[1:70,6],seriestype = :scatter,xlabel="interval-nr", label="D_combi",ylims=(-0.05,0.1))
plot!(1:70,info_noPrim[1:70,6],seriestype = :scatter,xlabel="interval-nr", label="D_noPrim",ylims=(-0.05,0.1))
plot!(1:70,info_full[1:70,6],seriestype = :scatter,xlabel="interval-nr", label="D_full",ylims=(-0.05,0.1))
plot!(1:70,info_effective[1:70,4],seriestype = :scatter,xlabel="interval-nr", label="D_eff",ylims=(-0.05,0.1))

####################################################################################
#    x-axis: interval-nr   y-axis: D
####################################################################################
plot(info_combi[1:70,13],info_combi[1:70,4],seriestype = :scatter,xlabel="G2", label="G1_combi",ylims=(-0.005,0.005))
plot!(info_combi[1:70,13],info_noPrim[1:70,4],seriestype = :scatter,xlabel="nadir", label="D_noPrim",ylims=(-0.00005,0.00005))
plot!(info_full[1:70,3],info_full[1:70,5],seriestype = :scatter,xlabel="G2", label="G1_full")
plot!(info_combi[1:70,13],info_effective[1:70,4],seriestype = :scatter,xlabel="interval-nr", label="D_eff")



####################################################################################
#    x-axis: model   y-axis: boxplot
####################################################################################

boxpl=Array{Float64,2}(undef,70,4)
boxpl[:,1]=info_combi[1:70,15]
boxpl[:,2]=info_noPrim[1:70,15]
boxpl[:,3]=info_full[1:70,15]
boxpl[:,4]=info_effective[1:70,10]
boxplot(["combi" "noPrim" "full" "effective"], boxpl, leg = false)



for i in 1:70
    if boxpl[i,1]==NaN
        boxpl[i,1]=10000000000
    end
end
for i in 1:70
    if boxpl[i,2]==NaN
        boxpl[i,2]=10000000000
    end
end
for i in 1:70
    if boxpl[i,1]==NaN
        boxpl[i,1]=10000000000
    end
end
for i in 1:70
    if boxpl[i,1]==NaN
        boxpl[i,1]=10000000000
    end
end
=#
#= ############### plot everything  combi-smooth
for hour in 40:50
    global turnOn=convert(Int64,combis70[hour,1])
    global posOfdbcrossing=convert(Int64,combis70[hour,2])
    convert(Int64,combis70[hour,1])
    TSdp=Array{Float64,1}(undef,900)
    TSdpt=Array{Float64,1}(undef,900)
    TS=timeseries_fpest(convert(Int64,combis70[hour,2]),convert(Int64,combis70[hour,1]),combis70[hour,3:12])
    for i in 1:900
         TSdp[i]=combis70[hour,5]+combis70[hour,6]*(i-1)
         if i<posOfdbcrossing+turnOn
             TSdpt[i]=TSdp[i]
         else
             TSdpt[i]=TSdp[i]-combis70[hour,4]*TS[i,3]
         end
    end
    TSsigma = sqrt.(abs.(TS[:,2]))
    TSmuPlus = TS[:,1]+TSsigma[:]
    TSmuMinus = TS[:,1]-TSsigma[:]
    yupper=maximum([maximum(TSdp),maximum(TSdpt),0])
    ylower=minimum([minimum(TSdp),minimum(TSdpt),0])
    inform="g1= "*string(round(combis70[hour,3];digits=3))*"  g2= "*string(round(combis70[hour,4];digits=5))*"  D= "*string(round(combis70[hour,7];digits=3))*"  r= "*string(round(combis70[hour,6];digits=7))*"  L= "*string(round(combis70[hour,16];digits=0))*"  AIC= "*string(round((2*10-2*combis70[hour,16]);digits=0))
    ueberschrift="Combi-model-smooth   hour "*string(hour)
    #path=".\plots\first70\combi"*string(hour)
    p1=plot(0:899,[TS[:,1] TSmuPlus TSmuMinus fs[((hour-1)*3600+1):((hour-1)*3600+900)]],label=false,ylabel="omega",xaxis=false,title=ueberschrift)
    p2=plot(0:899,[TSdp TSdpt],label=["dP" "dP+2nd"],ylims=(ylower,yupper),annotations = (0,yupper,Plots.text(inform, :left,10)),xlabel="time in s")
    plot(p1,p2,layout = (2, 1))
    png("./plots/first70/combis"*string(hour))
end
hour=21
global turnOn=convert(Int64,combis70[hour,1])
global posOfdbcrossing=convert(Int64,combis70[hour,2])
convert(Int64,combis70[hour,1])
TSdp=Array{Float64,1}(undef,900)
TSdpt=Array{Float64,1}(undef,900)
TS=timeseries_fpest(convert(Int64,combis70[hour,2]),convert(Int64,combis70[hour,1]),combis70[hour,3:12])
for i in 1:900
     TSdp[i]=combis70[hour,5]+combis70[hour,6]*(i-1)
     if i<posOfdbcrossing+turnOn
         TSdpt[i]=TSdp[i]
     else
         TSdpt[i]=TSdp[i]-combis70[hour,4]*TS[i,3]
     end
end
TSsigma = sqrt.(abs.(TS[:,2]))
TSmuPlus = TS[:,1]+TSsigma[:]
TSmuMinus = TS[:,1]-TSsigma[:]
yupper=maximum([maximum(TSdp),maximum(TSdpt),0])
ylower=minimum([minimum(TSdp),minimum(TSdpt),0])
inform="g1= "*string(round(combis70[hour,3];digits=3))*"  g2= "*string(round(combis70[hour,4];digits=5))*"  D= "*string(round(combis70[hour,7];digits=3))*"  r= "*string(round(combis70[hour,6];digits=7))*"  L= "*string(round(combis70[hour,16];digits=0))*"  AIC= "*string(round((2*10-2*combis70[hour,16]);digits=0))
ueberschrift="Combi-model-smooth   hour "*string(hour)
#path=".\plots\first70\combi"*string(hour)
p1=plot(0:899,[TS[:,1] TSmuPlus TSmuMinus fs[((hour-1)*3600+1):((hour-1)*3600+900)]],label=false,ylabel="omega",xaxis=false,title=ueberschrift)
p2=plot(0:899,[TSdp TSdpt],label=["dP" "dP+2nd"],ylims=(ylower,yupper),annotations = (0,yupper,Plots.text(inform, :left,10)),xlabel="time in s")
plot(p1,p2,layout = (2, 1))
#png("./plots/first70/combi"*string(hour))
=#

############################################################################################
# sort out intervals with negative var and ntervals with large deviation from mov-av-model #
############################################################################################
function sort_out(mleres,fs) #gives a list of all intervals where mle worked (no NaNs) and where mle makes sense
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
    #goodones2
    return goods2=convert.(Int64,goodones2[1:j-1]),likelidiff
end
(goods2,likelidiff)=sort_out(mleres,fs)
Plots.plot(goods2,likelidiff[goods2],seriestype=:scatter)
goods2


####################################################
# pick out intervals with specific characteristics #
####################################################

# Intervals with high goodness of fit (better three quartiles)
j=1
Plots.plot(1:8760,likelidiff,seriestype=:scatter,ylims=(0,3000))
qL75=quantile(likelidiff[goods2],0.75)
goodfit_temp=Array{Int64,1}(undef,length(goods2))
for i in goods2
    if likelidiff[i]<qL75
        goodfit_temp[j]=i
        global j+=1
    end
end
goodfit=goodfit_temp[1:j-1]


# Intervals with high goodness of fit (better half of intervals)
j=1
Plots.plot(1:8760,likelidiff,seriestype=:scatter,ylims=(0,3000))
qL5=quantile(likelidiff[goods2],0.25)
goodfit05_temp=Array{Int64,1}(undef,length(goods2))
for i in goods2
    if likelidiff[i]<qL5
        goodfit05_temp[j]=i
        global j+=1
    end
end
goodfit05=goodfit05_temp[1:j-1]


# Intervals with control on
j=1
con_temp=Array{Int64,1}(undef,length(goods2))
for i in goods2
    if mleres[i,3]>0
        con_temp[j]=i
        global j+=1
    end
end
con=con_temp[1:j-1]


# Intervals with control off
j=1
coff_temp=Array{Int64,1}(undef,length(goods2))
for i in goods2
    if mleres[i,2]==0
        coff_temp[j]=i
        global j+=1
    end
end
coff=coff_temp[1:j-1]


# Intervals with large g1 (just where control on)
j=1
Plots.plot(con,mleres[con,3],seriestype=:scatter)
g1l25=quantile(mleres[con,3],0.25)
lg1_temp=Array{Int64,1}(undef,length(con))
for i in con
    if mleres[i,3]>g1l25
        lg1_temp[j]=i
        global j+=1
    end
end
lg1=lg1_temp[1:j-1]


# Intervals with large q (just where control on!!!)
j=1
Plots.plot(con,abs.(mleres[con,5]),seriestype=:scatter)
qcl25=quantile(abs.(mleres[con,5]),0.25)
lqc_temp=Array{Int64,1}(undef,length(con))
for i in con
    if abs(mleres[i,5])>qcl25
        lqc_temp[j]=i
        global j+=1
    end
end
lqc=lqc_temp[1:j-1]


# Intervals with large r  (just where control on!!!)
j=1
Plots.plot(con,abs.(mleres[con,6]),seriestype=:scatter)
rcl25=quantile(abs.(mleres[con,6]),0.25)
lrc_temp=Array{Int64,1}(undef,length(con))
for i in con
    if abs(mleres[i,6])>rcl25
        lrc_temp[j]=i
        global j+=1
    end
end
lrc=lrc_temp[1:j-1]


# Intervals with small tc  (just where control on!!!)
j=1
Plots.plot(con,abs.(mleres[con,2]),seriestype=:scatter)
stc75=45#quantile(abs.(mleres[con,2]),0.25)
stc_temp=Array{Int64,1}(undef,length(con))
for i in con
    if abs(mleres[i,2])<stc75
        stc_temp[j]=i
        global j+=1
    end
end
stc=stc_temp[1:j-1]
plot!(stc,mleres[stc,2],seriestype=:scatter)


# Intervals with small D
j=1
Plots.plot(coff,abs.(mleres[coff,7]),seriestype=:scatter)#,ylims=(2e-7,0.001))
sd=2e-7#quantile(abs.(mleres[con,2]),0.25)
sd_temp=Array{Int64,1}(undef,length(coff))
for i in coff
    if abs(mleres[i,7])<sd
        sd_temp[j]=i
        global j+=1
    end
end
sd=sd_temp[1:j-1]
plot!(sd,mleres[sd,7],seriestype=:scatter)




#= Intervals with large g2
j=1
con_temp=Array{Int64,1}(undef,length(goods2))
for i in goods2
    if mleres[i,3]>0
        con_temp[j]=i
        global j+=1
    end
end
con=con_temp[1:j-1]


# Intervals with large nadir
j=1
lnad_temp=Array{Int64,1}(undef,length(goods2))
for i in goods2
    if mleres[i,3]>0
        lnad_temp[j]=i
        global j+=1
    end
end
lnad=lnad_temp[1:j-1]


# Intervals with large deviation in beginning of the interval
j=1
lnadbeg_temp=Array{Int64,1}(undef,length(goods2))
for i in goods2
    if mleres[i,3]>0
        lnadbeg_temp[j]=i
        global j+=1
    end
end
lnadbeg=lnadbeg_temp[1:j-1]=#





################################# special criteria
nad_val2=intersect(goodfit,con)
j=1
nadoneone_temp=Array{Int64,1}(undef,length(nad_val2))
for i in nad_val2
    if abs((mleres[i,5].+(mleres[i,6].*mleres[i,2]))./mleres[i,3])<1 #&& abs((mleres[i,5].+(mleres[i,6].*mleres[i,2]))./mleres[i,3])>0.2 && mleres[i,15]>0.2
        nadoneone_temp[j]=i
        global j+=1
    end
end
nadoneone=nadoneone_temp[1:j-1]




#=OLD check, whether approx are valid:
plot((mleres[goods2,3].^2)./4,mleres[goods2,4],seriestype = :scatter,ylims=(0,0.0004),xlabel="g1^2/4",ylabel="g2",label=false)
plot!(0.0:0.00001:0.0004,0.0:0.00001:0.0004,label=false)
#plot!(0.0:0.001:0.04,0.125*(0.0:0.001:0.04).^2)
goodsmallstemp=zeros(length(goods2))   #then the sqrt approx is valid with 10% deviation at most
j=1
for h in goods2
    if mleres[h,3]<0.045 && mleres[h,4] < 0.2*0.25*(mleres[h,3])^2 && mleres[h,4] > 0.000005
    #if mleres[h,4] < 0.05*(mleres[h,3])^2 && mleres[h,15] > 0.25
            goodsmallstemp[j]=h
            global j+=1
    end
end
goodsmalls=convert.(Int64,goodsmallstemp[1:(j-1)])
plot!((mleres[goodsmalls,3].^2)./4,mleres[goodsmalls,4],seriestype = :scatter,ylims=(0,0.0004),xlabel="g1^2/4",label=false)
#plot!(mleres[goodsmalls,3],mleres[goodsmalls,4],seriestype = :scatter,ylims=(0,0.0004))



## pick out intervals with small g2 and small r
plot(goods2,mleres[goods2,4],seriestype = :scatter,ylims=(0,0.0005))
plot(goods2,mleres[goods2,6],seriestype = :scatter,ylims=(0,0.00001))
smallg2r_temp=zeros(length(goods2))   #then the sqrt approx is valid with 10% deviation at most
j=1
for h in goods2
    if mleres[h,4] < 0.00001 && mleres[h,6] < 2.5e-6 && mleres[h,4] > 0
            smallg2r_temp[j]=h
            global j+=1
    end
end
smallg2r=convert.(Int64,smallg2r_temp[1:(j-1)])


## pick out intervals with large r and large g1
plot(goods2,mleres[goods2,3],seriestype = :scatter,ylims=(0,0.04))
plot(goods2,abs.(mleres[goods2,6]),seriestype = :scatter,ylims=(0,0.000025))
largeg1r_temp=zeros(length(goods2))   #then the sqrt approx is valid with 10% deviation at most
j=1
for h in goods2
    if mleres[h,3] > 0.02 && abs(mleres[h,6]) > 5e-6
            largeg1r_temp[j]=h
            global j+=1
    end
end
largeg1r=convert.(Int64,largeg1r_temp[1:(j-1)])


## pick out intervals with large g1
plot(goods2,mleres[goods2,3],seriestype = :scatter,ylims=(0,0.04))
largeg1_temp=zeros(length(goods2))   #then the sqrt approx is valid with 10% deviation at most
j=1
for h in goods2
    if mleres[h,3] > 0.02# && mleres[h,4] < 0.05*(mleres[h,3]).^2
            largeg1_temp[j]=h
            global j+=1
    end
end
largeg1=convert.(Int64,largeg1_temp[1:(j-1)])




## prep data which is in goods2 && has large g2
hlargeg2_temp=zeros(length(mleres[:,1]),6)
j=1
for i in 2:length(mleres[:,1])
    if i in goods2 && mleres[i,4]>0.00001
        hlargeg2_temp[j,1:5]=mleres[i,3:7]
        global hlargeg2_temp[j,6]=i
        global j+=1
    end
end
hlargeg2=hlargeg2_temp[1:j-1,:]
global hlargeg2[:,6]=convert.(Int64,hlargeg2[:,6])
daytime_hlargeg2=convert.(Int64,mod.(hlargeg2[:,6],24))
for i in 1:length(daytime_hlargeg2)
    if daytime_hlargeg2[i]==0
        global daytime_hlargeg2[i]=24
    end
end
daytime_hlargeg2
hlargeg2=#



############################
#                          #
#     NOW PLOT THINGS      #
#                          #
############################

function plotInt(hour)
    global turnOn=convert(Int64,mleres[hour,1])
    global posOfdbcrossing=convert(Int64,mleres[hour,2])
    mleres[hour,8]
    TSdp=Array{Float64,1}(undef,1000)
    TSdpt=Array{Float64,1}(undef,1000)
    TSdp[1:100].=NaN
    TSdpt[1:100].=NaN
    TS=timeseries_fpest(convert(Int64,mleres[hour,2]),convert(Int64,mleres[hour,1]),mleres[hour,3:12])
    for i in 1:900
         TSdp[i+100]=mleres[hour,5]+mleres[hour,6]*(i-1)
         if i<posOfdbcrossing+turnOn
             TSdpt[i+100]=TSdp[i+100]
         else
             TSdpt[i+100]=TSdp[i+100]-mleres[hour,4]*TS[i,3]
         end
    end
    TSsigma = sqrt.(abs.(TS[:,2]))
    TSmuPlus = TS[:,1]+TSsigma[:]
    TSmuMinus = TS[:,1]-TSsigma[:]
    yupper=maximum([maximum(TSdp[101:1000]),maximum(TSdpt[101:1000]),0])
    ylower=minimum([minimum(TSdp[101:1000]),minimum(TSdpt[101:1000]),0])
    inform="g1= "*string(round(mleres[hour,3];digits=5))*"   g2= "*string(round(mleres[hour,4];digits=5))*"   D= "*string(round(mleres[hour,7];digits=3))*"   r= "*string(round(mleres[hour,6];digits=7))*"   L= "*string(round(mleres[hour,16];digits=0))*"   AIC= "*string(round((2*10-2*mleres[hour,16]);digits=0))
    ueberschrift="\$\\textrm{model \\ 3, \\ hour \\ }"*string(hour)*"\$"
    #path=".\plots\first70\combi"*string(hour)
    p1=Plots.plot(0:899,[TS[:,1] TSmuPlus TSmuMinus],color=[:blue :lightblue :lightblue],style=[:solid :dash :dash],label=false,ylabel=L"\omega",xaxis=false,title=ueberschrift)
    Plots.plot!(0:899,fs[((hour-1)*3600+1):((hour-1)*3600+900)],label=false,color=:black)
    p2=Plots.plot(-100:899,[TSdp TSdpt],color=[:blue :lightblue],label=[L"\Delta \textrm{P}" L"\Delta \textrm{P}+2\textrm{nd}"],ylims=(ylower,yupper),xlabel=L"\textrm{time \ in \ s}") #,annotations = (0,yupper,Plots.text(inform, :left,10))
    Plots.plot(p1,p2,layout = (2, 1))
end
#plot(plot(0:10; ribbon = (LinRange(0, 2, 11), LinRange(0, 1, 11))), plot(0:10; ribbon = 0:0.5:5), plot(0:10; ribbon = sqrt), plot(0:10; ribbon = 1))
hourp=121
plotInt(hourp)
lkj
#png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_atPeak/plots/intsWc/3_"*string(hourp))

#                              1           2          3      4    5 6 7    8      9       10       11       12   13 14       15         16
# info_combi/full/noPrim:   [turnOn,posOfdbcrossing,gamma1,gamma2,q,r,D,mu_w_0,mu_a_0,var_ww_0,var_aa_0,rho_aw_0,q,nadir,maxOfSmooth,loglikeli]

# 1)  q/gammma1 vs nadir

   ## #           # ###        #    #
 #    #           ##   #        #  # #
 #   ##           #              #      #
  ### #           #             # #   # #
      #     ##    #       ##    # #     #
      #     #             #      #      #


# 1a) (q+r*t_nad)/gamma1 vs nadir   (remark: just intervals with control on - otherwise we would get infinities)
#mleres[goodfit,3]
#abs.((mleres[goodfit,5].+(mleres[goodfit,6].*mleres[goodfit,2]))./mleres[goodfit,3])
Plots.plot(abs.((mleres[con,5].+(mleres[con,6].*mleres[con,2]))./mleres[con,3]),
    mleres[con,15],
    seriestype=:scatter,
    xlabel="\$|(\\textrm{q}+\\textrm{r}\\cdot \\textrm{t}_N)/\\gamma_1|\$",
    ylabel=L"\textrm{Nadir}",
    xlims=(0,1),
    ylims=(0,1),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=4,
    framestyle = :box,
    markeralpha=0.2
)
corspearman(abs.((mleres[con,5].+(mleres[con,6].*mleres[con,2]))./mleres[con,3]),mleres[con,15])
png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_atPeak/plots/validationNadir/3_NadScat")
# now mark q
#=plot(abs.((mleres[goodfit,5].+(mleres[goodfit,6].*mleres[goodfit,2]))./mleres[goodfit,3]),
    mleres[goodfit,15],
    marker_z=abs.(1000*mleres[goodfit,5]),
    seriestype=:scatter,
    xlabel="\$|(\\textrm{q}+\\textrm{r}\\cdot \\textrm{t}_N)/\\gamma_1|\$",
    ylabel="\$\\textrm{Nadir}\$",
    colorbar_title="\$|\\textrm{q}\\cdot10^3|\$",
    xlims=(0,1),
    ylims=(0,1),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)=#
ploval=density_plot((0.0,1),
    (0.0,1),
    abs.((mleres[goodfit,5].+(mleres[goodfit,6].*mleres[goodfit,2]))./mleres[goodfit,3]),
    mleres[goodfit,15],
    abs.(1000*mleres[goodfit,5]),
    27,
    20)
Plots.plot(ploval[1],
    ploval[2],
    marker_z=log.(ploval[3]),
    seriestype=:scatter,
    xlabel="\$|(\\textrm{q}+\\textrm{r}\\cdot \\textrm{t}_N)/\\gamma_1|\$",
    ylabel="\$\\textrm{Nadir}\$",
    colorbar_title="\$\\log(|\\textrm{q}\\cdot10^3|)\$",
    xlims=(0,1),
    ylims=(0,1),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99)
# mark r
#=plot(abs.((mleres[goodfit,5].+(mleres[goodfit,6].*mleres[goodfit,2]))./mleres[goodfit,3]),
    mleres[goodfit,15],
    marker_z=abs.(100000*mleres[goodfit,6]),
    seriestype=:scatter,
    xlabel="\$|(\\textrm{q}+\\textrm{r}\\cdot \\textrm{t}_N)/\\gamma_1|\$",
    ylabel="\$\\textrm{Nadir}\$",
    colorbar_title="\$|\\textrm{r}\\cdot10^5|\$",
    xlims=(0,1),
    ylims=(0,1),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)=#
ploval1=density_plot((0.0,1),
    (0.0,1),
    abs.((mleres[goodfit,5].+(mleres[goodfit,6].*mleres[goodfit,2]))./mleres[goodfit,3]),
    mleres[goodfit,15],
    abs.(100000*mleres[goodfit,6]),
    27,
    20)
Plots.plot(ploval1[1],
    ploval1[2],
    marker_z=log.(ploval1[3]),
    seriestype=:scatter,
    xlabel="\$|(\\textrm{q}+\\textrm{r}\\cdot \\textrm{t}_N)/\\gamma_1|\$",
    ylabel="\$\\textrm{Nadir}\$",
    colorbar_title="\$\\log(|\\textrm{r}\\cdot10^5|)\$",
    xlims=(0,1),
    ylims=(0,1),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99)
# mark g1
#=plot(abs.((mleres[goodfit,5].+(mleres[goodfit,6].*mleres[goodfit,2]))./mleres[goodfit,3]),
    mleres[goodfit,15],
    marker_z=abs.(1000*mleres[goodfit,3]),
    seriestype=:scatter,
    xlabel="\$|(\\textrm{q}+\\textrm{r}\\cdot \\textrm{t}_N)/\\gamma_1|\$",
    ylabel="\$\\textrm{Nadir}\$",
    colorbar_title="\$\\gamma_1\\cdot10^3\$",
    xlims=(0,1),
    ylims=(0,1),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)=#
ploval2=density_plot((0.0,1),
    (0.0,1),
    abs.((mleres[goodfit,5].+(mleres[goodfit,6].*mleres[goodfit,2]))./mleres[goodfit,3]),
    mleres[goodfit,15],
    abs.(1000*mleres[goodfit,3]),
    27,
    20)
Plots.plot(ploval2[1],
    ploval2[2],
    marker_z=log.(ploval2[3]),
    seriestype=:scatter,
    xlabel="\$|(\\textrm{q}+\\textrm{r}\\cdot \\textrm{t}_N)/\\gamma_1|\$",
    ylabel="\$\\textrm{Nadir}\$",
    colorbar_title="\$\\gamma_1\\cdot10^3\$",
    xlims=(0,1),
    ylims=(0,1),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99)
# observation: for very small values of q,r,g1 we have large deviations from expected line.
# This makes sense: If q is small, it is rather likeli, that g1 is small as well (Is it???--> test this! Idea:
#   slow deviation does not need a lot of primary control).
#   So the fraction q/g1 does not have to be small if q or g1 is small, however the relative error of the fraction
#   gets rather large, even for small deviations in q or g1.
# conclusion: 1) check whether q,r,g1 really become small "together"
#             2) look for intervals where q,r,g1 are small


### 1) check whether q,r,g1 really become small "together" (ust intervals with control on)
lqc
lrc
lg1
length(intersect(lqc,lg1))/length(lqc)
length(intersect(lrc,lg1))/length(lqc)
length(intersect(lqc,lrc))/length(lqc)
length(intersect(lqc,lrc,lg1))/length(lqc)
nad_val=intersect(goodfit,lqc,lrc,lg1)
length(nad_val)/length(lqc)


### 2) just consider intervals with q,r,g in upper three quartiles
Plots.plot(abs.((mleres[nad_val,5].+(mleres[nad_val,6].*mleres[nad_val,2]))./mleres[nad_val,3]),
    mleres[nad_val,15],
    #marker_z=abs.(1000*mleres[nad_val,7]),
    seriestype=:scatter,
    xlabel="\$|(\\textrm{q}+\\textrm{r}\\cdot \\textrm{t}_N)/\\gamma_1|\$",
    ylabel="\$\\textrm{Nadir}\$",
    #colorbar_title="\$\\gamma_1\\cdot10^3\$",
    xlims=(0,1),
    ylims=(0,1),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
nad_vallinfit=Polynomials.fit(abs.((mleres[nad_val,5].+(mleres[nad_val,6].*mleres[nad_val,2]))./mleres[nad_val,3]),mleres[nad_val,15],1)
Plots.plot!(0:1,nad_vallinfit(0:1),label=latexstring("\\textrm{lin.fit: }"*string(round(nad_vallinfit[1]; digits=2))*"x+"*string(round(nad_vallinfit[0]; digits=2))))
@. prop(x,p)=p[1]*x
nadpropfit=curve_fit(prop,abs.((mleres[nad_val,5].+(mleres[nad_val,6].*mleres[nad_val,2]))./mleres[nad_val,3]),mleres[nad_val,15],[1.0])
Plots.plot!(0:1,coef(nadpropfit)[1]*(0:1),label=latexstring("\\textrm{prop.fit: }"*string(round(coef(nadpropfit)[1]; digits=2))*"x"))

###
plot(abs.((mleres[nadoneone,5].+(mleres[nadoneone,6].*mleres[nadoneone,2]))./mleres[nadoneone,3]),
    mleres[nadoneone,15],
    #marker_z=abs.(1000*mleres[nadoneone,7]),
    seriestype=:scatter,
    xlabel="\$|(\\textrm{q}+\\textrm{r}\\cdot \\textrm{t}_N)/\\gamma_1|\$",
    ylabel="\$\\textrm{Nadir}\$",
    #title="\$\\textrm{q,r,}\\gamma_1\\textrm{ large}\$",
    xlims=(0,1),
    ylims=(0,1),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
nadoneonelinfit=polyfit(abs.((mleres[nadoneone,5].+(mleres[nadoneone,6].*mleres[nadoneone,2]))./mleres[nadoneone,3]),mleres[nadoneone,15],1)
plot!(0:1,nadoneonelinfit(0:1),label=latexstring("\\textrm{lin.fit: }"*string(round(nadoneonelinfit[1]; digits=2))*"x+"*string(round(nadoneonelinfit[0]; digits=2))))
@. prop(x,p)=p[1]*x
nadpropfit=curve_fit(prop,abs.((mleres[nadoneone,5].+(mleres[nadoneone,6].*mleres[nadoneone,2]))./mleres[nadoneone,3]),mleres[nadoneone,15],[1.0])
plot!(0:1,coef(nadpropfit)[1]*(0:1),label=latexstring("\\textrm{prop.fit: }"*string(round(coef(nadpropfit)[1]; digits=2))*"x"))

# observations: |(q+r*t)/g1| tends to be estimated a bit too large for small q,r,g1 OR: Nadir is estimated to small (smothed curve...)
# Idea aout this:  g1 maybe not well seperated from g2 and therefor we have too small g1? (Chack whether g2 tends to be too large!!!)


# 2)  D,vs something

    ####
    #   #
    #    #
    #    #
    #   #
    ####

# 2.1) calculate the variance(omega - omega_smoothed)
vari=zeros(length(mleres[:,1]))
vari2=zeros(length(mleres[:,1]))   #starting from turnon
vari3=zeros(length(mleres[:,1]))
vari4=zeros(length(mleres[:,1]))
smoothfs=Array{Float64,1}(undef,900)
smoothfs2=Array{Float64,1}(undef,900)
for h in goods2
    global turnOn=convert(Int64,mleres[h,1])
    global posOfdbcrossing=convert(Int64,mleres[h,2])
    TS=timeseries_fpest(convert(Int64,mleres[h,2]),convert(Int64,mleres[h,1]),mleres[h,3:12])
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
    global vari2[h]=var(fs[((h-1)*3600+maximum([1,convert(Int64,mleres[h,2])])):((h-1)*3600+900)].-smoothfs2[maximum([1,convert(Int64,mleres[h,2])]):900])
    global vari3[h]=var(fs[((h-1)*3600+maximum([1,convert(Int64,mleres[h,2])])):((h-1)*3600+900)].-TS[(maximum([1,convert(Int64,mleres[h,2])]):900),1])
    global vari4[h]=var(fs[((h-1)*3600+1):((h-1)*3600+900)].-TS[1:900,1])
end

Plots.plot((mleres[goods2,7].^2)./(2*mleres[goods2,3]),
    vari2[goods2],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    xlims=(0.0,1.5e-2),
    ylims=(0.0,0.005),
    label="\$\\textrm{estimated}\$",
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=1
)
Plots.plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$")
png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_atPeak/plots/validationD/3_DScat")

########
# 1a) control on, subtract mov average
########
Plots.plot((mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    xlims=(0.0,1.5e-2),
    ylims=(0.0,0.005),
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=1,
    label="\$\\textrm{control on}\$"
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$")
# color g1 + g0
plovald=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    abs.(1000*(mleres[con,3]+mleres[con,13])),
    27,
    20)
plot(plovald[1],
    plovald[2],
    marker_z=log.(plovald[3]),
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\log(10^3\\cdot(\\gamma_0+\\gamma_1))\$",
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color g0
plovald=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    abs.(1000*mleres[con,13]),
    27,
    20)
plot(plovald[1],
    plovald[2],
    marker_z=plovald[3],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10^3\\cdot\\gamma_0\$",
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color g1
plovald=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    abs.(1000*mleres[con,3]),
    27,
    20)
Plots.plot(plovald[1],
    plovald[2],
    marker_z=(plovald[3]),
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\log(10^3\\cdot\\gamma_1)\$",
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_atPeak/plots/validationD/3_DScat_g1Col")
# color D
plovald1=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    abs.(1000*mleres[con,7]),
    27,
    20)
Plots.plot(plovald1[1],
    plovald1[2],
    marker_z=plovald1[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10^3\\cdot\\textrm{D}\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_atPeak/plots/validationD/3_DScat_DCol")
#color goodnes of fit
plovald2=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    likelidiff[con]./10,
    27,
    20)
plot(plovald2[1],
    plovald2[2],
    marker_z=plovald2[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10^{-1}\\cdot\\Delta\\textrm{L}\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color g2
plovald3=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    abs.(10000*mleres[con,4]),
    27,
    20)
plot(plovald3[1],
    plovald3[2],
    marker_z=plovald3[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10^4\\cdot\\gamma_2\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color q
plovald4=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    abs.(10000*mleres[con,5]),
    27,
    20)
plot(plovald4[1],
    plovald4[2],
    marker_z=log.(plovald4[3]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\log(|10^4\\cdot\\textrm{q}|)\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color r
plovald5=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    abs.(10000000*mleres[con,6]),
    27,
    20)
plot(plovald5[1],
    plovald5[2],
    marker_z=plovald5[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$|10^7\\cdot\\textrm{r}|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color mu_0
plovald6=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    abs.(100*mleres[con,8]),
    27,
    20)
plot(plovald6[1],
    plovald6[2],
    marker_z=plovald6[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$|10^2\\cdot\\mu_0|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color abs(nadir-mu_0)
plovald6=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    10*abs.(mleres[con,16]-mleres[con,8]),
    27,
    20)
plot(plovald6[1],
    plovald6[2],
    marker_z=plovald6[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10\\cdot|\\textrm{nadir - }\\mu_0|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color nadir
plovald7=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    abs.(10*mleres[con,16]),
    27,
    20)
plot(plovald7[1],
    plovald7[2],
    marker_z=plovald7[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$|10\\cdot\\textrm{nadir}|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)

########
# 1b) control on, subtract mu_mle
########
Plots.plot((mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{Var(freq - }\\mu_{mle})\$",
    xlims=(0.0,1.5e-2),
    ylims=(0.0,0.005),
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=1,
    label="\$\\textrm{estimated}\$"
)
Plots.plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$")
png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_atPeak/plots/validationD/3_DScatMu")

# color g1 + g0
plovald=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    abs.(1000*(mleres[con,3]+mleres[con,13])),
    27,
    20)
plot(plovald[1],
    plovald[2],
    marker_z=log.(plovald[3]),
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\log(10^3\\cdot(\\gamma_0+\\gamma_1))\$",
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color g0
plovald=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    abs.(1000*mleres[con,13]),
    27,
    20)
plot(plovald[1],
    plovald[2],
    marker_z=plovald[3],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$10^3\\cdot\\gamma_0\$",
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color g1
plovald=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    abs.(1000*mleres[con,3]),
    27,
    20)
plot(plovald[1],
    plovald[2],
    marker_z=log.(plovald[3]),
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\log(10^3\\cdot\\gamma_1)\$",
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color D
plovald1=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    abs.(1000*mleres[con,7]),
    27,
    20)
Plots.plot(plovald1[1],
    plovald1[2],
    marker_z=plovald1[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$10^3\\cdot\\textrm{D}\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
#color goodnes of fit
plovald2=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    likelidiff[con]./10,
    27,
    20)
plot(plovald2[1],
    plovald2[2],
    marker_z=plovald2[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$10^{-1}\\cdot\\Delta\\textrm{L}\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color g2
plovald3=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    abs.(10000*mleres[con,4]),
    27,
    20)
plot(plovald3[1],
    plovald3[2],
    marker_z=plovald3[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$10^4\\cdot\\gamma_2\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color q
plovald4=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    abs.(10000*mleres[con,5]),
    27,
    20)
plot(plovald4[1],
    plovald4[2],
    marker_z=log.(plovald4[3]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\log(|10^4\\cdot\\textrm{q}|)\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color r
plovald5=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    abs.(10000000*mleres[con,6]),
    27,
    20)
plot(plovald5[1],
    plovald5[2],
    marker_z=plovald5[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$|10^7\\cdot\\textrm{r}|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color mu_0
plovald6=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    abs.(100*mleres[con,8]),
    27,
    20)
plot(plovald6[1],
    plovald6[2],
    marker_z=plovald6[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$|10^2\\cdot\\mu_0|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color abs(nadir-mu_0)
plovald6=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    10*abs.(mleres[con,16]-mleres[con,8]),
    27,
    20)
plot(plovald6[1],
    plovald6[2],
    marker_z=plovald6[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$10\\cdot|\\textrm{nadir - }\\mu_0|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color nadir
plovald7=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    abs.(10*mleres[con,16]),
    27,
    20)
plot(plovald7[1],
    plovald7[2],
    marker_z=plovald7[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$|10\\cdot\\textrm{nadir}|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)


########
# 2a) control off, subtract mov average
########
plot((mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari2[coff],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    xlims=(0.0,1.5e-2),
    ylims=(0.0,0.005),
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=1,
    label="\$\\textrm{cofftrol off}\$"
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$")
# color g0
plovald=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari2[coff],
    abs.(1000*mleres[coff,13]),
    27,
    20)
plot(plovald[1],
    plovald[2],
    marker_z=plovald[3],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10^3\\cdot\\gamma_0\$",
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color D
plovald1=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari2[coff],
    abs.(1000*mleres[coff,7]),
    27,
    20)
plot(plovald1[1],
    plovald1[2],
    marker_z=plovald1[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10^3\\cdot\\textrm{D}\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
#color goodnes of fit
plovald2=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari2[coff],
    likelidiff[coff]./10,
    27,
    20)
plot(plovald2[1],
    plovald2[2],
    marker_z=plovald2[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10^{-1}\\cdot\\Delta\\textrm{L}\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color q
plovald4=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari2[coff],
    abs.(10000*mleres[coff,5]),
    27,
    20)
plot(plovald4[1],
    plovald4[2],
    marker_z=log.(plovald4[3]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\log(|10^4\\cdot\\textrm{q}|)\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color r
plovald5=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari2[coff],
    abs.(10000000*mleres[coff,6]),
    27,
    20)
plot(plovald5[1],
    plovald5[2],
    marker_z=plovald5[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$|10^7\\cdot\\textrm{r}|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color mu_0
plovald6=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari2[coff],
    abs.(100*mleres[coff,8]),
    27,
    20)
plot(plovald6[1],
    plovald6[2],
    marker_z=plovald6[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$|10^2\\cdot\\mu_0|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color abs(nadir-mu_0)
plovald6=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari2[coff],
    10*abs.(mleres[coff,16]-mleres[coff,8]),
    27,
    20)
plot(plovald6[1],
    plovald6[2],
    marker_z=plovald6[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10\\cdot|\\textrm{nadir - }\\mu_0|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color nadir
plovald7=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari2[coff],
    abs.(10*mleres[coff,16]),
    27,
    20)
plot(plovald7[1],
    plovald7[2],
    marker_z=plovald7[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$|10\\cdot\\textrm{nadir}|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)

########
# 2b) cofftrol off, subtract mu_mle
########
plot((mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari3[coff],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    xlims=(0.0,1.5e-2),
    ylims=(0.0,0.005),
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=1,
    label="\$\\textrm{cofftrol off}\$"
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$")
# color g0
plovald=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari3[coff],
    abs.(1000*mleres[coff,13]),
    27,
    20)
plot(plovald[1],
    plovald[2],
    marker_z=plovald[3],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$10^3\\cdot\\gamma_0\$",
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color D
plovald1=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari3[coff],
    abs.(1000*mleres[coff,7]),
    27,
    20)
plot(plovald1[1],
    plovald1[2],
    marker_z=plovald1[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$10^3\\cdot\\textrm{D}\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
#color goodnes of fit
plovald2=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari3[coff],
    likelidiff[coff]./10,
    27,
    20)
plot(plovald2[1],
    plovald2[2],
    marker_z=plovald2[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$10^{-1}\\cdot\\Delta\\textrm{L}\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color q
plovald4=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari3[coff],
    abs.(10000*mleres[coff,5]),
    27,
    20)
plot(plovald4[1],
    plovald4[2],
    marker_z=log.(plovald4[3]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\log(|10^4\\cdot\\textrm{q}|)\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color r
plovald5=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari3[coff],
    abs.(10000000*mleres[coff,6]),
    27,
    20)
plot(plovald5[1],
    plovald5[2],
    marker_z=plovald5[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$|10^7\\cdot\\textrm{r}|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color mu_0
plovald6=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari3[coff],
    abs.(100*mleres[coff,8]),
    27,
    20)
plot(plovald6[1],
    plovald6[2],
    marker_z=plovald6[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$|10^2\\cdot\\mu_0|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color abs(nadir-mu_0)
plovald6=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari3[coff],
    10*abs.(mleres[coff,16]-mleres[coff,8]),
    27,
    20)
plot(plovald6[1],
    plovald6[2],
    marker_z=plovald6[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$10\\cdot|\\textrm{nadir - }\\mu_0|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color nadir
plovald7=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,13]),
    vari3[coff],
    abs.(10*mleres[coff,16]),
    27,
    20)
plot(plovald7[1],
    plovald7[2],
    marker_z=plovald7[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_0)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$|10\\cdot\\textrm{nadir}|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)





































########
# subtract mov. average CONTROL OFF
########
# a) all intervals that are in coff
plot!(coff,mleres[coff,7],seriestype=:scatter,ylims=(0.0,0.01))
plot(con,mleres[con,7],seriestype=:scatter)#,ylims=(0.0,0.001))



plot((mleres[coff,7]),
    vari2[coff],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    xlims=(0.0,1.5e-2),
    ylims=(0.0,0.005),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.3
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color g1
plovald=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,3]),
    vari2[coff],
    abs.(1000*mleres[coff,3]),
    27,
    20)
plot(plovald[1],
    plovald[2],
    marker_z=log.(plovald[3]),
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\log(10^3\\cdot\\gamma1)\$",
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color D
plovald1=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,3]),
    vari2[coff],
    abs.(1000*mleres[coff,7]),
    27,
    20)
plot(plovald1[1],
    plovald1[2],
    marker_z=plovald1[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10^3\\cdot\\textrm{D}\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
#color goodnes of fit
plovald2=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,3]),
    vari2[coff],
    likelidiff[coff]./10,
    27,
    20)
plot(plovald2[1],
    plovald2[2],
    marker_z=plovald2[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10^{-1}\\cdot\\Delta\\textrm{L}\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color g2
plovald3=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,3]),
    vari2[coff],
    abs.(10000*mleres[coff,4]),
    27,
    20)
plot(plovald3[1],
    plovald3[2],
    marker_z=plovald3[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10^4\\cdot\\gamma_2\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color q
plovald4=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,3]),
    vari2[coff],
    abs.(10000*mleres[coff,5]),
    27,
    20)
plot(plovald4[1],
    plovald4[2],
    marker_z=log.(plovald4[3]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\log(|10^4\\cdot\\textrm{q}|)\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color r
plovald5=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,3]),
    vari2[coff],
    abs.(10000000*mleres[coff,6]),
    27,
    20)
plot(plovald5[1],
    plovald5[2],
    marker_z=plovald5[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$|10^7\\cdot\\textrm{r}|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color mu_0
plovald6=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,3]),
    vari2[coff],
    abs.(100*mleres[coff,8]),
    27,
    20)
plot(plovald6[1],
    plovald6[2],
    marker_z=plovald6[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$|10^2\\cdot\\mu_0|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color abs(nadir-mu_0)
plovald6=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,3]),
    vari2[coff],
    10*abs.(mleres[coff,15]-mleres[coff,8]),
    27,
    20)
plot(plovald6[1],
    plovald6[2],
    marker_z=plovald6[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10\\cdot|\\textrm{nadir - }\\mu_0|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color nadir
plovald7=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,3]),
    vari2[coff],
    abs.(10*mleres[coff,15]),
    27,
    20)
plot(plovald7[1],
    plovald7[2],
    marker_z=plovald7[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$|10\\cdot\\textrm{nadir}|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color nadir
plovald8=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[coff,7].^2)./(2*mleres[coff,3]),
    vari2[coff],
    mleres[coff,2],
    27,
    20)
plot(plovald8[1],
    plovald8[2],
    marker_z=plovald8[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\textrm{t}_c\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)


########
# subtract mov. average CONTROL ON
########
# a) all intervals that are in con
plot((mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.3
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color g1
#=plot((mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    marker_z=abs.(1000*mleres[con,3]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\gamma1\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)=#
plovald=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    abs.(1000*mleres[con,3]),
    27,
    20)
plot(plovald[1],
    plovald[2],
    marker_z=log.(plovald[3]),
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\log(10^3\\cdot\\gamma1)\$",
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color D
plovald1=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    abs.(1000*mleres[con,7]),
    27,
    20)
plot(plovald1[1],
    plovald1[2],
    marker_z=plovald1[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10^3\\cdot\\textrm{D}\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
#color goodnes of fit
plovald2=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    likelidiff[con]./10,
    27,
    20)
plot(plovald2[1],
    plovald2[2],
    marker_z=plovald2[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10^{-1}\\cdot\\Delta\\textrm{L}\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color g2
plovald3=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    abs.(10000*mleres[con,4]),
    27,
    20)
plot(plovald3[1],
    plovald3[2],
    marker_z=plovald3[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10^4\\cdot\\gamma_2\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color q
plovald4=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    abs.(10000*mleres[con,5]),
    27,
    20)
plot(plovald4[1],
    plovald4[2],
    marker_z=log.(plovald4[3]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\log(|10^4\\cdot\\textrm{q}|)\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color r
plovald5=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    abs.(10000000*mleres[con,6]),
    27,
    20)
plot(plovald5[1],
    plovald5[2],
    marker_z=plovald5[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$|10^7\\cdot\\textrm{r}|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color mu_0
plovald6=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    abs.(100*mleres[con,8]),
    27,
    20)
plot(plovald6[1],
    plovald6[2],
    marker_z=plovald6[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$|10^2\\cdot\\mu_0|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color abs(nadir-mu_0)
plovald6=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    10*abs.(mleres[con,15]-mleres[con,8]),
    27,
    20)
plot(plovald6[1],
    plovald6[2],
    marker_z=plovald6[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10\\cdot|\\textrm{nadir - }\\mu_0|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color nadir
plovald7=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    abs.(10*mleres[con,15]),
    27,
    20)
plot(plovald7[1],
    plovald7[2],
    marker_z=plovald7[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$|10\\cdot\\textrm{nadir}|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color nadir
plovald8=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    mleres[con,2],
    27,
    20)
plot(plovald8[1],
    plovald8[2],
    marker_z=plovald8[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\textrm{t}_c\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)



# b) all intervals with t_c smaller than 45 (to reduce the impact of the
#    difference between the domain when comparing the variance-estimation
#    between mle and mean-model)
plot((mleres[stc,7].^2)./(2*mleres[stc,3]),
    vari2[stc],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$\\textrm{var(freq - mean})\$",
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.3
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color g1
plovald=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[stc,7].^2)./(2*mleres[stc,3]),
    vari2[stc],
    abs.(1000*mleres[stc,3]),
    27,
    20)
plot(plovald[1],
    plovald[2],
    marker_z=log.(plovald[3]),
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\log(10^3\\cdot\\gamma1)\$",
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color D
plovald1=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[stc,7].^2)./(2*mleres[stc,3]),
    vari2[stc],
    abs.(1000*mleres[stc,7]),
    27,
    20)
plot(plovald1[1],
    plovald1[2],
    marker_z=plovald1[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10^3\\cdot\\textrm{D}\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
#color goodnes of fit
plovald2=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[stc,7].^2)./(2*mleres[stc,3]),
    vari2[stc],
    likelidiff[stc]./10,
    27,
    20)
plot(plovald2[1],
    plovald2[2],
    marker_z=plovald2[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10^{-1}\\cdot\\Delta\\textrm{L}\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color g2
plovald3=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[stc,7].^2)./(2*mleres[stc,3]),
    vari2[stc],
    abs.(10000*mleres[stc,4]),
    27,
    20)
plot(plovald3[1],
    plovald3[2],
    marker_z=plovald3[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10^4\\cdot\\gamma_2\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color q
plovald4=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[stc,7].^2)./(2*mleres[stc,3]),
    vari2[stc],
    abs.(10000*mleres[stc,5]),
    27,
    20)
plot(plovald4[1],
    plovald4[2],
    marker_z=log.(plovald4[3]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\log(|10^4\\cdot\\textrm{q}|)\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color r
plovald5=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[stc,7].^2)./(2*mleres[stc,3]),
    vari2[stc],
    abs.(10000000*mleres[stc,6]),
    27,
    20)
plot(plovald5[1],
    plovald5[2],
    marker_z=plovald5[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$|10^7\\cdot\\textrm{r}|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color mu_0
plovald6=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[stc,7].^2)./(2*mleres[stc,3]),
    vari2[stc],
    abs.(100*mleres[stc,8]),
    27,
    20)
plot(plovald6[1],
    plovald6[2],
    marker_z=plovald6[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$|10^2\\cdot\\mu_0|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color abs(nadir-mu_0)
plovald6=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[stc,7].^2)./(2*mleres[stc,3]),
    vari2[stc],
    10*abs.(mleres[stc,15]-mleres[stc,8]),
    27,
    20)
plot(plovald6[1],
    plovald6[2],
    marker_z=plovald6[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$10\\cdot|\\textrm{nadir - }\\mu_0|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color nadir
plovald7=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[stc,7].^2)./(2*mleres[stc,3]),
    vari2[stc],
    abs.(10*mleres[stc,15]),
    27,
    20)
plot(plovald7[1],
    plovald7[2],
    marker_z=plovald7[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$|10\\cdot\\textrm{nadir}|\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)
# color nadir
plovald8=density_plot((0.0,0.013),
    (0.0,0.005),
    (mleres[stc,7].^2)./(2*mleres[stc,3]),
    vari2[stc],
    mleres[stc,2],
    27,
    20)
plot(plovald8[1],
    plovald8[2],
    marker_z=plovald8[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\textrm{t}_c\$",
    seriestype=:scatter,
    xlims=(0.0,0.013),
    ylims=(0.0,0.005),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.013,0:0.0001:0.013,label="\$\\textrm{expected}\$",color=:green)



# a)   best 3 quartiles
goodfitcon=intersect(con,goodfit)
plot((mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="var from trun-on-pos (mov. av. subtracted)",
    xlims=(0.0,0.008),
    ylims=(0.0,0.004),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfitcon,7].^2)./(2*mleres[goodfitcon,3]),
    vari2[goodfitcon],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="var from trun-on-pos (mov. av. subtracted)",
    xlims=(0.0,0.008),
    ylims=(0.0,0.004),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfitcon,7].^2)./(2*mleres[goodfitcon,3]),
    vari2[goodfitcon],
    marker_z=abs.(1000*mleres[goodfitcon,3]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="var from trun-on-pos (mov. av. subtracted)",
    colorbar_title="\$\\gamma1\$",
    seriestype=:scatter,
    xlims=(0.0,0.008),
    ylims=(0.0,0.004),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfitcon,7].^2)./(2*mleres[goodfitcon,3]),
    vari2[goodfitcon],
    marker_z=abs.(10000*mleres[goodfitcon,7]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="var from trun-on-pos (mov. av. subtracted)",
    colorbar_title="\$\\textrm{D}\$",
    seriestype=:scatter,
    xlims=(0.0,0.008),
    ylims=(0.0,0.004),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfitcon,7].^2)./(2*mleres[goodfitcon,3]),
    vari2[goodfitcon],
    marker_z=likelidiff[goodfitcon],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="var from trun-on-pos (mov. av. subtracted)",
    colorbar_title="\$\\textrm{goodnes of fit}\$",
    seriestype=:scatter,
    xlims=(0.0,0.008),
    ylims=(0.0,0.004),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfitcon,7].^2)./(2*mleres[goodfitcon,3]),
    vari2[goodfitcon],
    marker_z=abs.(1000*mleres[goodfitcon,4]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="var from trun-on-pos (mov. av. subtracted)",
    colorbar_title="\$\\gamma_2\$",
    seriestype=:scatter,
    xlims=(0.0,0.008),
    ylims=(0.0,0.004),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
k
# b) best 2 quartiles
plot((mleres[con,7].^2)./(2*mleres[con,3]),
    vari2[con],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    xlims=(0.0,0.008),
    ylims=(0.0,0.008),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfit05,7].^2)./(2*mleres[goodfit05,3]),
    vari2[goodfit05],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    xlims=(0.0,0.008),
    ylims=(0.0,0.008),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfit05,7].^2)./(2*mleres[goodfit05,3]),
    vari2[goodfit05],
    marker_z=abs.(1000*mleres[goodfit05,3]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\gamma1\$",
    seriestype=:scatter,
    xlims=(0.0,0.008),
    ylims=(0.0,0.008),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfit05,7].^2)./(2*mleres[goodfit05,3]),
    vari2[goodfit05],
    marker_z=abs.(10000*mleres[goodfit05,7]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\textrm{D}\$",
    seriestype=:scatter,
    xlims=(0.0,0.008),
    ylims=(0.0,0.008),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfit05,7].^2)./(2*mleres[goodfit05,3]),
    vari2[goodfit05],
    marker_z=likelidiff[goodfit05],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\textrm{goodnes of fit}\$",
    seriestype=:scatter,
    xlims=(0.0,0.008),
    ylims=(0.0,0.008),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfit05,7].^2)./(2*mleres[goodfit05,3]),
    vari2[goodfit05],
    marker_z=abs.(10000*mleres[goodfit05,5]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\textrm{q}\$",
    seriestype=:scatter,
    xlims=(0.0,0.008),
    ylims=(0.0,0.008),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfit05,7].^2)./(2*mleres[goodfit05,3]),
    vari2[goodfit05],
    marker_z=abs.(1000*mleres[goodfit05,6]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - mean})\$",
    colorbar_title="\$\\textrm{r}\$",
    seriestype=:scatter,
    xlims=(0.0,0.008),
    ylims=(0.0,0.008),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfit05,7].^2)./(2*mleres[goodfit05,3]),
    vari2[goodfit05],
    marker_z=abs.(1000*mleres[goodfit05,3]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$\\textrm{var from trun-on-pos (mov. av. subtracted)}\$",
    seriestype=:scatter,
    colorbar_title="\$\\gamma_2\\cdot 10^3\$",
    xlims=(0.0,0.008),
    ylims=(0.0,0.008),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
k

#######
# now subtract \mu_{MLE}
#######
plot((mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    xlims=(0.0,0.0221),
    ylims=(0.0,0.15),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.3
)
plot!(0:0.0001:0.0221,0:0.0001:0.0221,label="\$\\textrm{expected}\$",color=:green)
# color g1
#=plot((mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    marker_z=abs.(1000*mleres[con,3]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\gamma1\$",
    seriestype=:scatter,
    xlims=(0.0,0.0221),
    ylims=(0.0,0.15),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)=#
plovaldmu=density_plot((0.0,0.0221),
    (0.0,0.15),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    abs.(1000*mleres[con,3]),
    27,
    20)
plot(plovaldmu[1],
    plovaldmu[2],
    marker_z=log.(plovaldmu[3]),
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\log(10^3\\cdot\\gamma1)\$",
    xlims=(0.0,0.0221),
    ylims=(0.0,0.15),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99)
plot!(0:0.0001:0.0221,0:0.0001:0.0221,label="\$\\textrm{expected}\$",color=:green)
# color D
plovaldmu1=density_plot((0.0,0.0221),
    (0.0,0.15),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    abs.(1000*mleres[con,7]),
    27,
    20)
plot(plovaldmu1[1],
    plovaldmu1[2],
    marker_z=plovaldmu1[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$10^3\\cdot\\textrm{D}\$",
    seriestype=:scatter,
    xlims=(0.0,0.0221),
    ylims=(0.0,0.15),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.0221,0:0.0001:0.0221,label="\$\\textrm{expected}\$",color=:green)
#color goodnes of fit
plovaldmu2=density_plot((0.0,0.0221),
    (0.0,0.15),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    likelidiff[con]./10,
    27,
    20)
plot(plovaldmu2[1],
    plovaldmu2[2],
    marker_z=plovaldmu2[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$10^{-1}\\cdot\\Delta\\textrm{L}\$",
    seriestype=:scatter,
    xlims=(0.0,0.0221),
    ylims=(0.0,0.15),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.0221,0:0.0001:0.0221,label="\$\\textrm{expected}\$",color=:green)
# color g2
plovaldmu3=density_plot((0.0,0.0221),
    (0.0,0.15),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    abs.(10000*mleres[con,4]),
    27,
    20)
plot(plovaldmu3[1],
    plovaldmu3[2],
    marker_z=plovaldmu3[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$10^4\\cdot\\gamma_2\$",
    seriestype=:scatter,
    xlims=(0.0,0.0221),
    ylims=(0.0,0.15),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.0221,0:0.0001:0.0221,label="\$\\textrm{expected}\$",color=:green)
# color q
plovaldmu4=density_plot((0.0,0.0221),
    (0.0,0.15),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    abs.(10000*mleres[con,5]),
    27,
    20)
plot(plovaldmu4[1],
    plovaldmu4[2],
    marker_z=log.(plovaldmu4[3]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\log(|10^4\\cdot\\textrm{q}|)\$",
    seriestype=:scatter,
    xlims=(0.0,0.0221),
    ylims=(0.0,0.15),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.0221,0:0.0001:0.0221,label="\$\\textrm{expected}\$",color=:green)
# color r
plovaldmu5=density_plot((0.0,0.0221),
    (0.0,0.15),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    abs.(10000000*mleres[con,6]),
    27,
    20)
plot(plovaldmu5[1],
    plovaldmu5[2],
    marker_z=plovaldmu5[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$|10^7\\cdot\\textrm{r}|\$",
    seriestype=:scatter,
    xlims=(0.0,0.0221),
    ylims=(0.0,0.15),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.0221,0:0.0001:0.0221,label="\$\\textrm{expected}\$",color=:green)
# color mu_0
plovaldmu6=density_plot((0.0,0.0221),
    (0.0,0.15),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    abs.(100*mleres[con,8]),
    27,
    20)
plot(plovaldmu6[1],
    plovaldmu6[2],
    marker_z=plovaldmu6[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$|10^2\\cdot\\mu_0|\$",
    seriestype=:scatter,
    xlims=(0.0,0.0221),
    ylims=(0.0,0.15),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.0221,0:0.0001:0.0221,label="\$\\textrm{expected}\$",color=:green)
# color abs(nadir-mu_0)
plovaldmu6=density_plot((0.0,0.0221),
    (0.0,0.15),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    10*abs.(mleres[con,15]-mleres[con,8]),
    27,
    20)
plot(plovaldmu6[1],
    plovaldmu6[2],
    marker_z=plovaldmu6[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$10\\cdot|\\textrm{nadir - }\\mu_0|\$",
    seriestype=:scatter,
    xlims=(0.0,0.0221),
    ylims=(0.0,0.15),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.0221,0:0.0001:0.0221,label="\$\\textrm{expected}\$",color=:green)
# color nadir
plovaldmu7=density_plot((0.0,0.0221),
    (0.0,0.15),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    abs.(10*mleres[con,15]),
    27,
    20)
plot(plovaldmu7[1],
    plovaldmu7[2],
    marker_z=plovaldmu7[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$|10\\cdot\\textrm{nadir}|\$",
    seriestype=:scatter,
    xlims=(0.0,0.0221),
    ylims=(0.0,0.15),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.0221,0:0.0001:0.0221,label="\$\\textrm{expected}\$",color=:green)
# color nadir
plovaldmu8=density_plot((0.0,0.0221),
    (0.0,0.15),
    (mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    mleres[con,2],
    27,
    20)
plot(plovaldmu8[1],
    plovaldmu8[2],
    marker_z=plovaldmu8[3],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\textrm{t}_c\$",
    seriestype=:scatter,
    xlims=(0.0,0.0221),
    ylims=(0.0,0.15),
    label=false,
    markershape=:rect,
    markerstrokewidth = 0,
    markersize=10,
    framestyle = :box,
    markeralpha=0.99
)
plot!(0:0.0001:0.0221,0:0.0001:0.0221,label="\$\\textrm{expected}\$",color=:green)



# a)   best 3 quartiles
plot((mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    xlims=(0.0,0.02),
    ylims=(0.0,0.02),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfitcon,7].^2)./(2*mleres[goodfitcon,3]),
    vari3[goodfitcon],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    xlims=(0.0,0.02),
    ylims=(0.0,0.02),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfitcon,7].^2)./(2*mleres[goodfitcon,3]),
    vari3[goodfitcon],
    marker_z=abs.(1000*mleres[goodfitcon,3]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\gamma1\$",
    seriestype=:scatter,
    xlims=(0.0,0.02),
    ylims=(0.0,0.02),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfitcon,7].^2)./(2*mleres[goodfitcon,3]),
    vari3[goodfitcon],
    marker_z=abs.(10000*mleres[goodfitcon,7]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\textrm{D}\$",
    seriestype=:scatter,
    xlims=(0.0,0.02),
    ylims=(0.0,0.02),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfitcon,7].^2)./(2*mleres[goodfitcon,3]),
    vari3[goodfitcon],
    marker_z=likelidiff[goodfitcon],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\textrm{goodnes of fit}\$",
    seriestype=:scatter,
    xlims=(0.0,0.02),
    ylims=(0.0,0.02),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfitcon,7].^2)./(2*mleres[goodfitcon,3]),
    vari3[goodfitcon],
    marker_z=abs.(10000*mleres[goodfitcon,5]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\textrm{q}\$",
    seriestype=:scatter,
    xlims=(0.0,0.02),
    ylims=(0.0,0.02),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfitcon,7].^2)./(2*mleres[goodfitcon,3]),
    vari3[goodfitcon],
    marker_z=abs.(1000*mleres[goodfitcon,6]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\textrm{r}\$",
    seriestype=:scatter,
    xlims=(0.0,0.02),
    ylims=(0.0,0.02),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfitcon,7].^2)./(2*mleres[goodfitcon,3]),
    vari3[goodfitcon],
    marker_z=abs.(1000*mleres[goodfitcon,4]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$\\textrm{var from trun-on-pos (mov. av. subtracted)}\$",
    seriestype=:scatter,
    colorbar_title="\$\\gamma_2\\cdot 10^3\$",
    xlims=(0.0,0.02),
    ylims=(0.0,0.02),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)


# b) best 2 quartiles
plot((mleres[con,7].^2)./(2*mleres[con,3]),
    vari3[con],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    xlims=(0.0,0.02),
    ylims=(0.0,0.02),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfit05,7].^2)./(2*mleres[goodfit05,3]),
    vari3[goodfit05],
    seriestype=:scatter,
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    xlims=(0.0,0.02),
    ylims=(0.0,0.02),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfit05,7].^2)./(2*mleres[goodfit05,3]),
    vari3[goodfit05],
    marker_z=abs.(1000*mleres[goodfit05,3]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\gamma1\$",
    seriestype=:scatter,
    xlims=(0.0,0.02),
    ylims=(0.0,0.02),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfit05,7].^2)./(2*mleres[goodfit05,3]),
    vari3[goodfit05],
    marker_z=abs.(10000*mleres[goodfit05,7]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\textrm{D}\$",
    seriestype=:scatter,
    xlims=(0.0,0.02),
    ylims=(0.0,0.02),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfit05,7].^2)./(2*mleres[goodfit05,3]),
    vari3[goodfit05],
    marker_z=likelidiff[goodfit05],
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\textrm{goodnes of fit}\$",
    seriestype=:scatter,
    xlims=(0.0,0.02),
    ylims=(0.0,0.02),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfit05,7].^2)./(2*mleres[goodfit05,3]),
    vari3[goodfit05],
    marker_z=abs.(10000*mleres[goodfit05,5]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\textrm{q}\$",
    seriestype=:scatter,
    xlims=(0.0,0.02),
    ylims=(0.0,0.02),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfit05,7].^2)./(2*mleres[goodfit05,3]),
    vari3[goodfit05],
    marker_z=abs.(1000*mleres[goodfit05,6]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$ \\textrm{var(freq - }\\mu_{mle})\$",
    colorbar_title="\$\\textrm{r}\$",
    seriestype=:scatter,
    xlims=(0.0,0.02),
    ylims=(0.0,0.02),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)
plot((mleres[goodfit05,7].^2)./(2*mleres[goodfit05,3]),
    vari3[goodfit05],
    marker_z=abs.(1000*mleres[goodfit05,4]),
    xlabel="\$\\textrm{D}^2/(2\\gamma_1)\$",
    ylabel="\$\\textrm{var from trun-on-pos (mov. av. subtracted)}\$",
    seriestype=:scatter,
    colorbar_title="\$\\gamma_2\\cdot 10^3\$",
    xlims=(0.0,0.02),
    ylims=(0.0,0.02),
    label=false,
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.99
)


coeficient of determination xi^2
l




#= 2.2)  D,vs D of KRAMERS-MOYAL (start with gamma1 estimation after turning control on.)
#large range on omega-axis
sep=40
Dkmval=zeros(sep+1,2)
Dkmres=zeros(672)
for h in goods2
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
    #plot(0:899,fs[((h-1)*3600+1):((h-1)*3600+900)].-smoothfs2,label=false)
    Dkm=zeros(sep+1,2)
    for i in convert(Int64,mleres[h,2]+1):900
        if -0.1<(fs[((h-1)*3600+i)]-smoothfs2[i])&&(fs[((h-1)*3600+i)]-smoothfs2[i])<0.1
            Dkm[convert(Int64,floor((fs[((h-1)*3600+i)]-smoothfs2[i]+0.1)*sep/0.2+1)),1]+=(fs[((h-1)*3600+i+1)]-fs[((h-1)*3600+i)])^2
            Dkm[convert(Int64,floor((fs[((h-1)*3600+i)]-smoothfs2[i]+0.1)*sep/0.2+1)),2]+=1
        end
    end
    j=1
    for i in 1:sep+1
        if Dkm[i,2]>4
            global Dkmval[j,:]=[(Dkm[i,1]./Dkm[i,2]),(0.2/sep)*(i-1)-0.1]
            j+=1
        end
    end
    Dkmval2=Dkmval[1:(j-1),:]
    #plot(Dkmval2[:,2],Dkmval2[:,1],seriestype = :scatter,label=false)
    fitf=polyfit(Dkmval2[:,2],Dkmval2[:,1],1)[0]
    function leastsq(par)
        sum((par.-Dkmval2[:,1]).^2)
    end
    fitf2=optimize(leastsq, [fitf-(1000*fitf)], [fitf+(3000*fitf)], [fitf], Fminbox(LBFGS())).minimizer
    #yval=zeros(j-1)
    #for i in 1:(j-1)
    #    yval[i]=fitf2[1]
    #end
    #plot!(Dkmval2[:,2],yval)
    Dkmres[h]=sqrt(fitf2[1]*2)
end
plot(Dkmres[goods2],mleres[goods2,7],seriestype = :scatter, xlabel="D_km",ylabel="D_MLE",label=false,ylims=(0.0,0.025),xlims=(0.0,0.025))
plot!(Dkmres[smallg2r],mleres[smallg2r,7],seriestype = :scatter, xlabel="D_km",ylabel="D_MLE",label=false,ylims=(0.0,0.025),xlims=(0.0,0.025))
plot!(Dkmres[largeg1r],mleres[largeg1r,7],seriestype = :scatter, xlabel="D_km",ylabel="D_MLE",label=false,ylims=(0.0,0.025),xlims=(0.0,0.025))
fitDlarge=polyfit(Dkmres[goods2],mleres[goods2,7],1)
fitDlarge2grsmall=polyfit(Dkmres[smallg2r],mleres[smallg2r,7],1)
fitDlarge2grlarge=polyfit(Dkmres[largeg1r],mleres[largeg1r,7],1)
plot!((0.00:0.001:0.025),fitDlarge(0.00:0.001:0.025))
plot!((0.00:0.001:0.025),fitDlarge2grsmall(0.00:0.001:0.025))
plot!((0.00:0.001:0.025),fitDlarge2grlarge(0.00:0.001:0.025))



# 2.3)  D,vs D of Kramers-Moyal (start with gamma1 estimation after turning control on.)
# now we just take a smaller interval around 50 hz
sep=20
Dkmval=zeros(sep+1,2)
Dkmres3=zeros(672)
for h in goods2
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
    #plot(0:899,fs[((h-1)*3600+1):((h-1)*3600+900)].-smoothfs2,label=false)
    Dkm=zeros(sep+1,2)
    for i in convert(Int64,mleres[h,2]+1):900
        if -0.025<(fs[((h-1)*3600+i)]-smoothfs2[i])&&(fs[((h-1)*3600+i)]-smoothfs2[i])<0.025
            Dkm[convert(Int64,floor((fs[((h-1)*3600+i)]-smoothfs2[i]+0.025)*(sep/0.05)+1)),1]+=(fs[((h-1)*3600+i+1)]-fs[((h-1)*3600+i)])^2
            Dkm[convert(Int64,floor((fs[((h-1)*3600+i)]-smoothfs2[i]+0.025)*(sep/0.05)+1)),2]+=1
        end
    end
    j=1
    for i in 1:sep+1
        if Dkm[i,2]>4
            global Dkmval[j,:]=[(Dkm[i,1]./Dkm[i,2]),(0.05/sep)*(i-1)-0.025]
            j+=1
        end
    end
    Dkmval2=Dkmval[1:(j-1),:]
    #plot(Dkmval2[:,2],Dkmval2[:,1],seriestype = :scatter,label=false)
    fitf=polyfit(Dkmval2[:,2],Dkmval2[:,1],1)[0]
    function leastsq(par)
        sum((par.-Dkmval2[:,1]).^2)
    end
    fitf2=optimize(leastsq, [fitf-(1000*fitf)], [fitf+(3000*fitf)], [fitf], Fminbox(LBFGS())).minimizer
    #yval=zeros(j-1)
    #for i in 1:(j-1)
    #    yval[i]=fitf2[1]
    #end
    #plot!(Dkmval2[:,2],yval)
    Dkmres3[h]=sqrt(fitf2[1]*2)
end
plot(Dkmres3[goods2],mleres[goods2,7],seriestype = :scatter, xlabel="D_km",ylabel="D_MLE",label=false,ylims=(0.0,0.025),xlims=(0.0,0.025))
plot!(Dkmres3[smallg2r],mleres[smallg2r,7],seriestype = :scatter, xlabel="D_km",ylabel="D_MLE",label=false,ylims=(0.0,0.025),xlims=(0.0,0.025))
fitDsmall=polyfit(Dkmres3[goods2],mleres[goods2,7],1)
fitDsmall2grsmall=polyfit(Dkmres3[smallg2r],mleres[smallg2r,7],1)
plot!((0.00:0.001:0.025),fitDsmall(0.00:0.001:0.025))
plot!((0.00:0.001:0.025),fitDsmall2grsmall(0.0:0.001:0.025))
## remark:
# larger gives better result here: slope is nearly 1, y-axis intersection is closer to zero.
# taking just the intervals where gamma2 and r is small should give better results, since mu is varying slowly in this case



# 2.2)  g1,vs g1 of Kramers-Moyal (start with gamma1 estimation after turning control on.)
gamma1=zeros(672)
kmval=zeros(41,2)
for h in goods2
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
    #plot(0:899,fs[((h-1)*3600+1):((h-1)*3600+900)].-smoothfs2,label=false)
    km=zeros(41,2)
    for i in convert(Int64,mleres[h,2]+1):900
        if -0.1<(fs[((h-1)*3600+i)]-smoothfs2[i])&&(fs[((h-1)*3600+i)]-smoothfs2[i])<0.1
            km[convert(Int64,floor((fs[((h-1)*3600+i)]-smoothfs2[i]+0.1)*40/0.2+1)),1]+=(fs[((h-1)*3600+i+1)]-fs[((h-1)*3600+i)])
            km[convert(Int64,floor((fs[((h-1)*3600+i)]-smoothfs2[i]+0.1)*40/0.2+1)),2]+=1
        end
    end
    j=1
    for i in 1:41
        if km[i,1]!=0
            global kmval[j,:]=[(km[i,1]./km[i,2]),(0.2/40)*(i-1)-0.1]
            j+=1
        end
    end
    kmval2=kmval[1:(j-1),:]
    #plot(kmval2[:,2],kmval2[:,1],seriestype = :scatter,label=false)
    global gamma1[h]=polyfit(kmval2[:,2],kmval2[:,1],1)[1]
    #plot!(kmval2[:,2],fitf(kmval2[:,2]))
end
plot(-gamma1[goods2],mleres[goods2,3],seriestype = :scatter, xlabel="g1_km",ylabel="g1_MLE",label=false)#,ylims=(0.0,0.04),xlims=(0.0,0.013))
goods3temp=zeros(length(goods2)) # just look at intervals where kramer moyal gives pos result and control is on
j=1
for i in goods2
    if -gamma1[i]>0 && mleres[i,3]!=0.0
            global goods3temp[j]=i
            global j+=1
    end
end
goods3=convert.(Int64,goods3temp[1:j-1])
plot(-gamma1[goods3],mleres[goods3,3],seriestype = :scatter, xlabel="g1_km",ylabel="g1_MLE",label=false)#,ylims=(0.0,0.04),xlims=(0.0,0.013))
plot(-gamma1[smallg2r],mleres[smallg2r,3],seriestype = :scatter, xlabel="g1_km",ylabel="g1_MLE",label=false)#,ylims=(0.0,0.04),xlims=(0.0,0.013))
plot(-gamma1[largeg1r],mleres[largeg1r,3],seriestype = :scatter, xlabel="g1_km",ylabel="g1_MLE",label=false)#,ylims=(0.0,0.04),xlims=(0.0,0.013))
polyfit(-gamma1[goods3],mleres[goods3,3],1)
polyfit(-gamma1[smallg2r],mleres[smallg2r,3],1)
polyfit(-gamma1[largeg1r],mleres[largeg1r,3],1)
=#

















#3) exp decay

    #    #
     #  # #
      #     ##
     # #   #  #
     # #     #
      #     #
           ####


#g2a=zeros(672)
#rb=zeros(672)
g2c=zeros(700)
g2cf=zeros(700)
#g2f=zeros(700)
#g2m=zeros(700)
for h in goodsmalls
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
    #plot(0:899,fs[((h-1)*3600+1):((h-1)*3600+900)],label=false)
    #plot!(0:899,smoothfs2,label=false)
    #@. modela(x, p) = p[1]*exp(-x*p[2])+p[3]
    #global g2a[h]=mleres[h,6]/coef(curve_fit(modela,400:900,smoothfs2[399:899],[0.0,0.01,0.0]))[3]
    #plot!(0:899,modela(0:899,tfit),label=false,ylims=(-0.6,0.6))
    #@. modelb(x, p) = p[1]*exp(-x*(mleres[h,4]/mleres[h,3]))+p[2]/mleres[h,4]
    #global rb[h]=coef(curve_fit(modelb,400:900,smoothfs2[399:899],[0.0,0.0]))[2]
    #plot!(0:899,modelb(0:899,tfitb),label=false,ylims=(-0.6,0.6))
    @. modelc(x, p) = p[1]*exp(-x*(p[2]/mleres[h,3]))+mleres[h,6]/p[2]
    global g2c[h]=coef(curve_fit(modelc,convert(Int64,maximum([mleres[h,2],200])):900,smoothfs2[convert(Int64,(maximum([mleres[h,2],200])-1)):899],[0.0,0.0001]))[2]
    #@. modelcf(x, p) = p[1]*exp(-0.5*x*(mleres[h,3]-sqrt.(mleres[h,3].^2(-(p[2].*4)))))+mleres[h,6]/p[2]
    #lower_p = [-10000.0; 0.0]
    #upper_p = [10000.0; (mleres[h,3]^2)/5]
    #global g2cf[h]=coef(curve_fit(modelcf,convert(Int64,maximum([mleres[h,2],200])):900,smoothfs2[convert(Int64,(maximum([mleres[h,2],200])-1)):899],[0.0,0.00001], lower = lower_p, upper = upper_p))[2]
    #plot!(0:899,modelc(0:899,tfitc),label=false,ylims=(-0.6,0.6))
    #@. modelf(x, p) = p[1]*exp(-x*p[2])+p[3]*exp(-x*p[4])+p[5]
    #global g2f[h]=mleres[h,6]/coef(curve_fit(modelf,convert(Int64,mleres[h,2]+1):900,smoothfs2[(convert(Int64,mleres[h,2]+1)):900],[0.0,0.0001,0.0,0.0,0.0]))[5]
    #plot!(0:899,modelf(0:899,tfitf),label=false,ylims=(-0.6,0.6))
    #global g2m[h]=mleres[h,6]/smoothfs2[900]
end
#plot(g2f[goodsmalls],mleres[goodsmalls,4],seriestype = :scatter,xlims=(0,0.00025),ylims=(0,0.00025))
#plot(g2a[goodsmalls],mleres[goodsmalls,4],seriestype = :scatter,xlims=(0,0.00025),ylims=(0,0.00025))
plot(g2c[goodsmalls],mleres[goodsmalls,4],seriestype = :scatter, label=false,xlims=(0,0.0001),ylims=(0,0.0001),xlabel="g2_fit",ylabel="g2_MLE")
#plot(g2m[goodsmalls],mleres[goodsmalls,4],seriestype = :scatter,xlims=(0,0.00025),ylims=(0,0.00025))

#plot(g2f[largeg1],mleres[largeg1,4],seriestype = :scatter, label=false,xlims=(0,0.001),ylims=(0,0.001))#,xlims=(0,0.000),ylims=(0,0.00075))
#plot(g2a[largeg1],mleres[largeg1,4],seriestype = :scatter, label=false,xlims=(0,0.001),ylims=(0,0.001))#,xlims=(0,0.00025),ylims=(0,0.00075))
plot(g2c[largeg1],mleres[largeg1,4],seriestype = :scatter, label=false,xlims=(0,0.0001),ylims=(0,0.0001),xlabel="g2_fit",ylabel="g2_MLE")
#plot(g2cf[largeg1],mleres[largeg1,4],seriestype = :scatter, label=false,xlims=(0,0.0002),ylims=(0,0.0002),xlabel="g2_fit",ylabel="g2_MLE")

#fit model c with g1 large
fitc_temp=zeros(length(goodsmalls))   #then the sqrt approx is valid with 10% deviation at most
j=1
for h in goodsmalls
    if g2c[h] < 0.0001
            global fitc_temp[j]=h
            global j+=1
    end
end
fitc=convert.(Int64,fitc_temp[1:(j-1)])
##plot!(g2c[fitc],mleres[fitc,4],seriestype = :scatter, label=false,xlims=(0,0.001),ylims=(0,0.001))#,xlims=(0,0.00025),ylims=(0,0.00075))
#fit_c=polyfit(g2c[fitc],mleres[fitc,4],1)
#plot!(0:0.00001:0.001,fit_c(0:0.00001:0.001))
@. prop(x,p)=p[1]*x
g2prop=coef(curve_fit(prop,g2c[fitc],mleres[fitc,4],[1.0]))[1]
plot!(0.0:0.001:0.008,prop(0.0:0.001:0.008,g2prop),label="0.857*x")
#comment:


#=h=21
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
plot(0:899,fs[((h-1)*3600+1):((h-1)*3600+900)],label=false)
plot!(0:899,smoothfs2,label=false)
@. modela(x, p) = p[1]*exp(-x*p[2])+p[3]
tfit=coef(curve_fit(modela,400:900,smoothfs2[399:899],[0.0,0.01,0.0]))
plot!(0:899,modela(0:899,tfit),label=false,ylims=(-0.6,0.6))

@. modelb(x, p) = p[1]*exp(-x*(mleres[h,4]/mleres[h,3]))+p[2]/mleres[h,4]
tfitb=coef(curve_fit(modelb,400:900,smoothfs2[399:899],[0.0,0.0]))
plot!(0:899,modelb(0:899,tfitb),label=false,ylims=(-0.6,0.6))

@. modelc(x, p) = p[1]*exp(-x*(p[2]/mleres[h,3]))+mleres[h,6]/p[2]
tfitc=coef(curve_fit(modelc,400:900,smoothfs2[399:899],[0.0,0.0001]))
plot!(0:899,modelc(0:899,tfitc),label=false,ylims=(-0.6,0.6))

@. modelcf(x, p) = p[1]*exp(-0.5*x*(mleres[h,3]-sqrt.(mleres[h,3].^2(-(p[2].*4)))))+mleres[h,6]/p[2]
lower_p = [-10000.0; 0.0]
upper_p = [10000.0; (mleres[h,3]^2)/5]
global g2cf[h]=coef(curve_fit(modelcf,convert(Int64,maximum([mleres[h,2],200])):900,smoothfs2[convert(Int64,(maximum([mleres[h,2],200])-1)):899],[0.0,0.000001], lower = lower_p, upper = upper_p))[2]

curve_fit(model, x, y, p; lower = lower_p, upper = upper_p)

@. modelf(x, p) = p[1]*exp(-x*p[2])+p[3]*exp(-x*p[4])+p[5]
tfitf=coef(curve_fit(modelf,convert(Int64,mleres[h,2]):900,smoothfs2[(convert(Int64,mleres[h,2])-1):899],[0.0,0.0001,0.0,0.0,0.0]))
plot!(0:899,modelf(0:899,tfitf),label=false,ylims=(-0.6,0.6))
mleres[h,6]/tfitf[5]
=#






#######################################
# plot dayly profile of mle parameters #
########################################

fid=h5open("../data/other/CE_data_collected_by_johannes/input_actual_timezone_berlin.h5","r")
dgr=fid["df"]
#println(names(dgr))
data=read(dgr["block0_values"])
sces=read(dgr["block0_items"])
sce1=read(dgr["axis0"])
sce2=read(dgr["axis1"])
sce2=unix2datetime.(sce2[:]/1000000000)
#sce2[35065+0:(35065+364*24)]
sce2[26305+0:(26305+364*24+23)]
startd=26305
sces[20]


function byHour(tval,yval)
    datas_=Array{AbstractArray,1}(undef,24)
    hour_vals=zeros(length(tval))
    for i in 1:24
        count=1
        for j in 1:length(tval)
            if tval[j]==i && isnan(yval[j])==false
                hour_vals[count]=yval[j]
                count+=1
            end
        end
        datas_[i]=hour_vals[1:count-1]
    end
    datas_
end

# q
Plots.plot(mod.(goods2,24).+1,mleres[goods2,5],
    seriestype=:scatter,
    xlabel="\$\\textrm{hour}\$",
    ylabel="\$ \\textrm{q}\$",
    #xlims=(0.0,1.5e-2),
    #ylims=(0.0,0.005),
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.1,
    label=false#"\$\\textrm{control on}\$"
)
Plots.boxplot(byHour(mod.(goods2,24).+1,mleres[goods2,5]),
    #seriestype=:scatter,
    xlabel="\$\\textrm{hour}\$",
    ylabel="\$ \\textrm{q}\$",
    #xlims=(0.0,1.5e-2),
    #ylims=(0.0,0.005),
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=1,
    color="lightblue",
    label=false#"\$\\textrm{mean}\$"
)
Plots.plot!(1:24,mean.(byHour(mod.(goods2,24).+1,mleres[goods2,5])),
    seriestype=:scatter,
    xlabel="\$\\textrm{hour}\$",
    ylabel="\$ \\textrm{q}\$",
    #xlims=(0.0,1.5e-2),
    #ylims=(0.0,0.005),
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=1,
    color="red",
    label="\$\\textrm{mean}\$"
)

png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_comparison/plots/hist12.png")

# -r
Plots.plot(mod.(goods2,24).+1,-mleres[goods2,6],
    seriestype=:scatter,
    xlabel="\$\\textrm{hour}\$",
    ylabel="\$ \\textrm{-r}\$",
    #xlims=(0.0,1.5e-2),
    #ylims=(0.0,0.005),
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=0.1,
    label=false#"\$\\textrm{control on}\$"
)
Plots.boxplot(byHour(mod.(goods2,24).+1,-mleres[goods2,6]),
    #seriestype=:scatter,
    xlabel="\$\\textrm{hour}\$",
    ylabel="\$ \\textrm{-r}\$",
    #xlims=(0.0,1.5e-2),
    #ylims=(0.0,0.005),
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=1,
    color="lightblue",
    label=false#"\$\\textrm{mean}\$"
)
Plots.plot!(1:24,mean.(byHour(mod.(goods2,24).+1,-mleres[goods2,6])),
    seriestype=:scatter,
    xlabel="\$\\textrm{hour}\$",
    ylabel="\$ \\textrm{-r}\$",
    #xlims=(0.0,1.5e-2),
    #ylims=(0.0,0.005),
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=1,
    color="red",
    label="\$\\textrm{mean}\$"
)
png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_comparison/plots/r.png")


# -r G_sync
Plots.boxplot(byHour(mod.(goods2,24).+1,-mleres[goods2,6].*data[20,26304 .+ goods2]),
    #seriestype=:scatter,
    xlabel="\$\\textrm{hour}\$",
    ylabel="\$ \\textrm{-r}\\cdot G_{sync}\$",
    #xlims=(0.0,1.5e-2),
    ylims=(-7.5,4.5),
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=1,
    color="lightblue",
    label=false#"\$\\textrm{mean}\$"
)
Plots.plot!(1:24,mean.(byHour(mod.(goods2,24).+1,-mleres[goods2,6].*data[20,26304 .+ goods2])),
    seriestype=:scatter,
    xlabel="\$\\textrm{hour}\$",
    ylabel="\$ \\textrm{-r}\\cdot G_{sync}\$",
    #xlims=(0.0,1.5e-2),
    #ylims=(0.0,0.005),
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=1,
    color="red",
    label="\$\\textrm{mean}\$"
)
png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_comparison/plots/rG.png")


# G_sync
Plots.boxplot(byHour(mod.(goods2,24).+1,data[20,26304 .+ goods2]),
    #seriestype=:scatter,
    xlabel="\$\\textrm{hour}\$",
    ylabel="\$ \\textrm{-r}\\cdot G_{sync}\$",
    #xlims=(0.0,1.5e-2),
    ylims=(100000,330000),
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=1,
    color="lightblue",
    label=false#"\$\\textrm{mean}\$"
)
Plots.plot!(1:24,mean.(byHour(mod.(goods2,24).+1,data[20,26304 .+ goods2])),
    seriestype=:scatter,
    xlabel="\$\\textrm{hour}\$",
    ylabel="\$G_{sync}\$",
    #xlims=(0.0,1.5e-2),
    #ylims=(0.0,0.005),
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=1,
    color="red",
    label="\$\\textrm{mean}\$"
)
png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_comparison/plots/G.png")


#load_ramp
Plots.boxplot(byHour(mod.(goods2,24).+1,data[21,26304 .+ goods2]),
    #seriestype=:scatter,
    xlabel="\$\\textrm{hour}\$",
    ylabel="\$ \\textrm{load \\ ramp}\$",
    #xlims=(0.0,1.5e-2),
    #ylims=(100000,330000),
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=1,
    color="lightblue",
    label=false#"\$\\textrm{mean}\$"
)
Plots.plot!(1:24,mean.(byHour(mod.(goods2,24).+1,data[21,26304 .+ goods2])),
    seriestype=:scatter,
    xlabel="\$\\textrm{hour}\$",
    ylabel="\$\\textrm{load \\ ramp}\$",
    #xlims=(0.0,1.5e-2),
    #ylims=(0.0,0.005),
    markershape=:circle,
    markerstrokewidth = 0,
    markersize=3,
    framestyle = :box,
    markeralpha=1,
    color="red",
    label="\$\\textrm{mean}\$"
)
png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/model_comparison/plots/load_ramp.png")








print(sce1[:])



###############################################################################
fid=h5open("../data/other/CE_data_collected_by_johannes/input_actual_timezone_berlin.h5","r")
dgr=fid["df"]
#println(names(dgr))
data=read(dgr["block0_values"])
sces=read(dgr["block0_items"])
sce1=read(dgr["axis0"])
sce2=read(dgr["axis1"])
sce2=unix2datetime.(sce2[:]/1000000000)
#sce2[35065+0:(35065+364*24)]
sce2[26305+0:(26305+364*24+23)]
startd=26305
sces[20]

# first check that goods2 does not start with hour 1
goods2
mleres

#### take just intervals with control on: #####
j=1
con_temp=Array{Int64,1}(undef,length(goods2))
for i in goods2
    if mleres[i,3]>0
        con_temp[j]=i
        global j+=1
    end
end
con=con_temp[1:j-1]

RoCoFLong=Array{Float64,1}(undef,8760)
smoothfsLong=Array{Float64,1}(undef,401)
save_mleres=Array{Float64,2}(undef,8760,19) #17: D^2/(2*g1),    18: (q+r*t_N)/g1,    19: RoCof with longer timespan
for h in 1:8760
    if h in con
        save_mleres[h,1:16]=mleres[h,:]
        save_mleres[h,17]=(mleres[h,7]^2)/(2*mleres[h,3])
        save_mleres[h,18]=(mleres[h,5]+mleres[h,6]*mleres[h,2])/mleres[h,3]
        for i in -200:200
            global smoothfsLong[i+201]=sum(fs[(((h-1)*3600)+(i-50)):(((h-1)*3600)+(i+50))])/100
        end
        RoCoFLong[h]=Polynomials.fit(1:61,smoothfsLong[180:240],1)[1]
        save_mleres[h,19]=RoCoFLong[h]
    else
        for j in 1:19
            save_mleres[h,j]=NaN
        end
    end
end
save_mleres[:,3]=save_mleres[:,3].*1e04
save_mleres[:,4]=save_mleres[:,4].*1e07 # g1 abschneiden/durch Inertia teilen
save_mleres[:,5]=save_mleres[:,5].*1e05
save_mleres[:,6]=save_mleres[:,6].*1e09
save_mleres[:,7]=save_mleres[:,7].*1e03
save_mleres[:,17]=save_mleres[:,17].*1e04
for i in con
    if save_mleres[i,17]>100
        for j in 1:19
            save_mleres[i,j]=NaN
        end
    end
    if save_mleres[i,4]>5000
        for j in 1:19
            save_mleres[i,j]=NaN
        end
    end
    if save_mleres[i,3]>399
        for j in 1:19
            save_mleres[i,j]=NaN
        end
    end
end

##### check #####
mleres
save_mleres
Plots.plot(goods2[1:80],save_mleres[goods2[1:80],2])
Plots.plot(con[1:80],save_mleres[con[1:80],2])
Plots.plot(mod.(1:8760,24),save_mleres[:,17].*data[20,26305+0:(26305+364*24+23)],seriestype=:scatter,markerstrokewidth = 0,alpha=0.1)#,ylims=(0,1000))
Plots.plot(mod.(1:8760,24),save_mleres[:,17],seriestype=:scatter,markerstrokewidth = 0,alpha=0.1)
Plots.plot(1:240,save_mleres[1:240,3],markerstrokewidth = 0,alpha=1)
#ylims=(0,0.017),
##### save #####
Tables.table(save_mleres)
#writedlm( "/Users/raphaelbiertz/Documents/masterarbeit/coding/ML/prepared_data/mle/mleres18_re9.csv",save_mleres)
CSV.write("/Users/raphaelbiertz/Documents/masterarbeit/coding/ML/prepared_data/mle/mleres18_re9ADDDe4.csv",Tables.table(save_mleres))
