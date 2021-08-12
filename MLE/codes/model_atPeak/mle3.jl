using LinearAlgebra
using Polynomials
using Optim
using Distributions
using LsqFit
using Plots
using CSV
using DifferentialEquations
using DelimitedFiles
using Dates
using DataFrames

datas=CSV.File("../data/freq/Data_cleansed/TransnetBW/2019.csv",header=0)
fs=(datas[:,2].-50).*2*pi
fsbeg=Array{Float64,1}(undef,90)
smoothfs=Array{Float64,1}(undef,900)
smoothfs2=Array{Float64,1}(undef,900)
i_smoothfs=Array{Float64,1}(undef,300)
i_smoothfs2=Array{Float64,1}(undef,300)
i_smoothfs3=Array{Float64,1}(undef,300)
posOfdbcrossing=-1
var_ww_0 = 0.0
var_aa_0 = 0.0
rho_aw_0 = 0.0
D_guess = 0.0
minpos2=1
minpos=1
solRev=Array{Float64,2}(undef,900,3)
varOme=Array{Float64,1}(undef,900)
parguessRev=[0.02,0.0005,D_guess,var_ww_0,var_aa_0,rho_aw_0]
LB = [0.001,0.0,0.0,(var_ww_0/500), 0.0, -20.0]
UB = [0.04,1.0e-3,1.0,(var_ww_0*500), +120.0, +20.0]
parguessRes=[0.02,0.0005,D_guess,var_ww_0,var_aa_0,rho_aw_0]
q = 0.0
omega_0 = 0.0
par_guess1=[q,(q/900),omega_0,0.0,0.0,0.0]
LB1 = [q*1.2,0.0,omega_0-0.075,-320.0,0.0,0.0]
UB1 = [q*0.83,1.0,omega_0+0.075,320.0,0.0,0.0]
meanOme=Array{Float64,1}(undef,900)
sol=Array{Float64,2}(undef,900,2)
par_guess1Res=[q,(q/900),omega_0,0.0,0.0,0.0]

MLE = Array{Float64,1}(undef,10) #[gamma1,gamma2,q,r,D,mu_w_0,mu_a_0,var_ww_0,var_aa_0,rho_aw_0]
LB2 = [0.0, 0.0, -1.0,0.0,-1.0,-2.0,0.0, 50.0, 0.0, -20.0]
UB2 = [0.45,1.0e-3,0.0,1.0,1.0, 2.0,0.0, 50.0, +120.0, +20.0]
MLENoC = Array{Float64,1}(undef,8) #[gamma1,gamma2,q,r,D,mu_w_0,mu_a_0,var_ww_0,var_aa_0,rho_aw_0]
LB2NoC = [-1.0,0.0,-1.0,-2.0,0.0, 50.0, 0.0, -20.0]
UB2NoC = [0.0,1.0,1.0, 2.0,0.0, 50.0, +120.0, +20.0]
MS=Array{Float64,2}(undef,900,2)
sol4=Array{Float64,2}(undef,900,5)
sol5=Array{Float64,2}(undef,900,2)
MLEres = Array{Float64,1}(undef,10)
info=Array{Float64,2}(undef,8784,16)
t0=1000
turnOn=20
rocof=0

function dyn_moments1Rev!(dy,y,par,t)  #par will look like: [gamma1,gamma2,D]
    dy[1] = par[3]^2-(2*par[1]*y[1])-(2*par[2]*y[3])
    dy[2] = 2*y[3]
    dy[3] = -par[1]*y[3]-par[2]*y[2]+y[1]
end
function dyn_moments1_2Rev!(dy,y,par,t)  #par will look like: [gamma1,gamma2,D]
    dy[1] = par[3]^2-(2*par[1]*((t-posOfdbcrossing+1)/turnOn)*y[1])
    dy[2] = 2*y[3]
    dy[3] = -par[1]*((t-posOfdbcrossing+1)/turnOn)*y[3]+y[1]
end
function dyn_moments0Rev!(dy,y,par,t)  #par will look like: [gamma1,gamma2,D]
    dy[1] = par[3]^2
    dy[2] = 2*y[3]
    dy[3] = y[1]
end
function halfDetTimeseries_fpeRev(par) #par: [gamma1,gamma2,D,var_ww_0,var_aa_0,cov_wa_0]
    if posOfdbcrossing==0       #always in db-->control off
        so=solve(ODEProblem(dyn_moments0Rev!,[par[4],par[5],par[6]],(0.0,899.0),[par[1],par[2],par[3]]))
        for j in 1:900
            global solRev[j,:]=so(j-1.0)
        end
    else                             # posOfdbcrossing>0 (first inside db, then outside)
        so=solve(ODEProblem(dyn_moments0Rev!,[par[4],par[5],par[6]],(0.0,posOfdbcrossing-1.0),[par[1],par[2],par[3]]))
        for j in 1:posOfdbcrossing
            global solRev[j,:]=so(j-1.0)
        end
        so=solve(ODEProblem(dyn_moments1_2Rev!,[solRev[posOfdbcrossing,1],solRev[posOfdbcrossing,2],solRev[posOfdbcrossing,3]],(posOfdbcrossing-1.0,posOfdbcrossing+(turnOn-1.0)),[par[1],par[2],par[3]]))
        for j in posOfdbcrossing:(posOfdbcrossing+turnOn)
            global solRev[j,:]=so(j-1.0)
        end
        so=solve(ODEProblem(dyn_moments1Rev!,[solRev[(posOfdbcrossing+turnOn),1],solRev[(posOfdbcrossing+turnOn),2],solRev[(posOfdbcrossing+turnOn),3]],(posOfdbcrossing+(turnOn-1.0),899.0),[par[1],par[2],par[3]]))
        for j in (posOfdbcrossing+turnOn):900
            global solRev[j,:]=so(j-1.0)
        end
    end
    return solRev[:,1]  #out: array containing var(w)
end
function dyn_ode1!(dy,y,par,t) #par=[gamma1,gamma2,q,r]  y=[w,a]
    dy[1] = par[3]+(par[4]*t)-par[1]*y[1]-par[2]*y[2]
    dy[2] = y[1]
end
function dyn_ode1_2!(dy,y,par,t) #par=[gamma1,gamma2,q,r]  y=[w,a]
    dy[1] = par[3]+(par[4]*t)-par[1]*((t-posOfdbcrossing+1)/turnOn)*y[1]
    dy[2] = y[1]
end
function dyn_ode0!(dy,y,par,t) #par=[gamma1,gamma2,q,r]  y=[w,a]
    dy[1] = par[3]+(par[4]*t)
    dy[2] = y[1]
end
function timeseries_ode(par)
    if posOfdbcrossing==0       #always in db-->control off
        so=solve(ODEProblem(dyn_ode0!,[par[3],par[4]],(0.0,899.0),[par[5],par[6],par[1],par[2]]))
        for j in 1:900
            global sol[j,:]=so(j-1.0)
        end
    else                             # posOfdbcrossing>0 (first inside db, then outside)
        so=solve(ODEProblem(dyn_ode0!,[par[3],par[4]],(0.0,posOfdbcrossing-1),[par[5],par[6],par[1],par[2]]))
        for j in 1:posOfdbcrossing
            global sol[j,:]=so(j-1.0)
        end
        so=solve(ODEProblem(dyn_ode1_2!,[sol[posOfdbcrossing,1],sol[posOfdbcrossing,2]],(posOfdbcrossing-1.0,posOfdbcrossing+(turnOn-1.0)),[par[5],par[6],par[1],par[2]]))
        for j in posOfdbcrossing:(posOfdbcrossing+turnOn)
            global sol[j,:]=so(j-1.0)
        end
        so=solve(ODEProblem(dyn_ode1!,[sol[(posOfdbcrossing+turnOn),1],sol[(posOfdbcrossing+turnOn),2]],(posOfdbcrossing+(turnOn-1.0),899.0),[par[5],par[6],par[1],par[2]]))
        for j in (posOfdbcrossing+turnOn):900
            global sol[j,:]=so(j-1.0)
        end
    end
    return sol[:,1]
end
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
function timeseries_fpe(par)
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
function dyn_moments02NoC!(dy,y,par,t)  #par will look like: [q,r,D] WHY THIS FUNCTION? if we do not have control
    dy[1] = par[1]+(par[2]*t)
    dy[2] = y[1]
    dy[3] = par[3]^2
    dy[4] = 2*y[5]
    dy[5] = y[3]
end
function timeseries_fpeNoC(par)
    #always in db-->control off
    so=solve(ODEProblem(dyn_moments02NoC!,[par[4],par[5],par[6],par[7],par[8]],(0.0,899.0),par[1:3]))
    for j in 1:900
        global sol4[j,:]=so(j-1.0)
    end
    global sol5[:,1]=sol4[:,1]
    global sol5[:,2]=sol4[:,3]
    return sol5
end

# how to find the right starting point?

for h in 1:500
    if Second(DateTime(datas[(3600*(h-1)+1),1][1:10]*"T"*datas[(3600*(h-1)+1),1][12:19]))!=Second(0)
        ch=0
        for i in -20:20
            if Second(DateTime(datas[(3600*(h-1)+1+i),1][1:10]*"T"*datas[(3600*(h-1)+1+i),1][12:19]))==Second(0)
                ch=i
            end
        end
        global fs[(3600*(h-1)+1):(length(fs)-maximum([ch,0]))]=fs[(3600*(h-1)+1+ch):length(fs)+minimum([ch,0])]
        global datas[(3600*(h-1)+1):(length(fs)-maximum([ch,0])),:]=datas[(3600*(h-1)+1+ch):length(fs)+minimum([ch,0]),:]
    end
    badData=0
    for i in (3600*(h-1)+1):(3600*(h-1)+900)
        if (typeof(fs[i])==Float64 && isnan(fs[i])==false)==false # if data is missing or NaN set badData to 1
            badData=1
            break
        end
    end
    if badData!=1
        function halfDetLoglikeRev(par::AbstractArray) #par: [gamma1,gamma2,D,var_ww_0,var_aa_0,cov_wa_0]
            global varOme=halfDetTimeseries_fpeRev(par)
            global lnp = -sum((-1/2)*log.(2*pi*abs.(varOme))-(((fs[(3600*(h-1)+1):(3600*(h-1)+900)]-smoothfs2).^2)./(2*abs.(varOme))))
            if minimum(varOme)<1.0e-10
                global lnp=10000000000000000000000000
            end
            return lnp
        end
        function halfDetLoglikeODE(par::AbstractArray) #[q,r,omega_0,theta_0,gamma1,gamma2]
            global meanOme=timeseries_ode(par)
            global varOme=halfDetTimeseries_fpeRev(parguessRes)
            global lnp = -sum((-1/2)*log.(2*pi*abs.(varOme))-(((smoothfs2-meanOme).^2)./(2*abs.(varOme))))
            if minimum(varOme)<1.0e-10
                global lnp=10000000000000000000000000
            end
            return lnp
        end
        function loglike(par::AbstractArray)
            global MS=timeseries_fpe(par)
            global lnp = -sum((-1/2)*log.(2*pi*abs.(MS[:,2]))-(((fs[(3600*(h-1)+1):(3600*(h-1)+900)]-MS[:,1]).^2)./(2*abs.(MS[:,2]))))
            if minimum(MS[:,2])<1.0e-10
                global lnp=10000000000000000000000000
            end
            return lnp
        end
        function loglikeNoC(par::AbstractArray)
            global MS=timeseries_fpeNoC(par)
            global lnp = -sum((-1/2)*log.(2*pi*abs.(MS[:,2]))-(((fs[(3600*(h-1)+1):(3600*(h-1)+900)]-MS[:,1]).^2)./(2*abs.(MS[:,2]))))
            if minimum(MS[:,2])<1.0e-10
                global lnp=10000000000000000000000000
            end
            return lnp
        end
        try
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
            for i in 861:900
                global smoothfs2[i]=sum(smoothfs[(900-(2*(900-i))):900])/(2*(900-i))
            end
            for i in 880:900
                global smoothfs2[i]=smoothfs2[879]
            end
            global fsbeg[1:30].=fs[((h-1)*3600)+1]
            global fsbeg[31:90]=fs[((h-1)*3600+1):((h-1)*3600+60)]
            for i in 1:30
                global i_smoothfs2[i]=sum(fsbeg[i:(i+60)])/61
                global i_smoothfs[i]=sum(fsbeg[26+i:34+i])/9
            end
            for i in 31:300
                global i_smoothfs[i]=sum(fs[(((h-1)*3600)+(i-4)):(((h-1)*3600)+(i+4))])/9
                global i_smoothfs2[i]=sum(fs[(((h-1)*3600)+(i-30)):(((h-1)*3600)+(i+30))])/61
            end
            for i in 11:290
                i_smoothfs3[i]=sum(i_smoothfs2[(i-10):(i+10)])/21
            end
            global i_smoothfs3[1:10]=i_smoothfs2[1:10]
            global i_smoothfs3[291:300].=i_smoothfs3[290]
            if polyfit(5:15,i_smoothfs3[5:15],1)[1]>0
                for i in 11:299
                    if i_smoothfs3[i+1]<i_smoothfs3[i]
                        global minpos2=i
                        break
                    end
                end
                for i in maximum([(minpos2-25),1]):minimum([(minpos2+25),300])
                    if i_smoothfs[i+1]<i_smoothfs[i]
                        global minpos=i
                        break
                    end
                end
            else
                for i in 11:299
                    if i_smoothfs3[i+1]>i_smoothfs3[i]
                        global minpos2=i
                        break
                    end
                end
                for i in maximum([(minpos2-25),1]):minimum([(minpos2+25),300])
                    if i_smoothfs[i+1]>i_smoothfs[i]
                        global minpos=i
                        break
                    end
                end
            end
            if minpos>14
                global rocof=polyfit(1:minpos,i_smoothfs[1:minpos],1)[1]
                global turnOn=minimum([30,convert(Int64,abs(round((20/0.0045)*rocof)))])
            end
            #plot(-100:899,fs[((h-1)*3600+1-100):((h-1)*3600+900)])
            #plot!(0:299,i_smoothfs3)
            #plot!(0:299,i_smoothfs2)
            #plot!(0:299,i_smoothfs)
            #plot!(0:899,smoothfs2)
            #plot!(0:minpos-1,rocof(1:minpos))
            global posOfdbcrossing=maximum([1,convert(Int64,round(minpos-(turnOn/2)))])
            if turnOn<5
                global posOfdbcrossing = 0
            elseif minpos<15
                global posOfdbcrossing = 0
            elseif maximum(abs.(i_smoothfs[(minpos-10):(minpos+10)]))<0.13
                global posOfdbcrossing = 0
            end
            global var_ww_0 = (std(fs[((h-1)*3600+1):((h-1)*3600+900)]-smoothfs2))^2
            global var_aa_0 = var_ww_0/10
            global rho_aw_0 = var_ww_0
            global D_guess = sqrt(var_ww_0*2*0.02)
            if posOfdbcrossing==0
                global omega_0=i_smoothfs3[1]
                global MLENoC=[0.0,0.0,D_guess,omega_0,0.0,var_ww_0,var_aa_0,rho_aw_0]
                global LB2NoC = [-0.5,-0.05,0.0,omega_0-0.05,-10.0,(var_ww_0/500), 0.0, -20.0]
                global UB2NoC = [0.5, 0.05, 1.0,omega_0+0.05, 10.0, (var_ww_0*500), +120.0, +20.0]
                global MLEresNoC = optimize(loglikeNoC, LB2NoC, UB2NoC, MLENoC, Fminbox(LBFGS()),).minimizer
                global info[h,:]=[0.0,0.0,0.0,0.0,MLEresNoC[1],MLEresNoC[2],MLEresNoC[3],MLEresNoC[4],MLEresNoC[5],MLEresNoC[6],MLEresNoC[7],MLEresNoC[8],0,maximum(abs.(fs[(3600*(h-1)+1):(3600*(h-1)+900)])),maximum(abs.(smoothfs[1:450])),-loglikeNoC(MLEresNoC)]
                #[posOfdbcrossing,MLE,RoCof,nadir,MeanDevInFirstHalf,Loglike]
            else
                global parguessRev=[0.003,0.00005,D_guess,var_ww_0,var_aa_0,rho_aw_0]
                global LB = [0.001,0.0,0.0,(var_ww_0/500), 0.0, -20.0]
                global UB = [0.04,1.0e-3,1.0,(var_ww_0*500), +120.0, +20.0]
                global parguessRes = optimize(halfDetLoglikeRev, LB, UB, parguessRev, Fminbox(BFGS())).minimizer #Optim.Options(show_trace=true,iterations=100)
                #par: [gamma1,gamma2,D,var_ww_0,var_aa_0,cov_wa_0]
                #plot(0:899,fs[((h-1)*3600+1):((h-1)*3600+900)])
                #plot!(0:899,smoothfs2)
                #plot!(0:899,(smoothfs2+sqrt.(halfDetTimeseries_fpeRev(parguessRes))))
                #plot!(0:899,(smoothfs2-sqrt.(halfDetTimeseries_fpeRev(parguessRes))))
                par_temp = polyfit(1:posOfdbcrossing,i_smoothfs[1:posOfdbcrossing],1)
                #plot!(1:posOfdbcrossing,par_temp(1:posOfdbcrossing))
                global q = par_temp[1]
                global omega_0 = par_temp[0]
                global par_guess1=[q,(-q/600),omega_0,0.0,parguessRes[1],parguessRes[2]] #[q,r,omega_0,theta_0,gamma1,gamma2]
                if q<0.0
                    global LB1 = [q*1.01,-q*0.83/3000,omega_0-0.025,-10.0,0.0001,0.0]
                    global UB1 = [q*0.99,-q*1.2/300,omega_0+0.025,10.0,0.04,1.0e-3]
                else
                    global LB1 = [q*0.99,-q*1.2/300,omega_0-0.025,-10.0,0.0001,0.0]
                    global UB1 = [q*1.01,-q*0.83/3000,omega_0+0.025,10.0,0.04,1.0e-3]
                end
                global par_guess1Res = optimize(halfDetLoglikeODE, LB1, UB1, par_guess1, Fminbox(LBFGS())).minimizer #Optim.Options(show_trace=true,iterations=100)
                #plot(0:899,fs[((h-1)*3600+1):((h-1)*3600+900)])
                #plot!(0:899,smoothfs2)
                #plot!(0:899,timeseries_ode(par_guess1Res))
                #plot!(0:899,(timeseries_ode(par_guess1Res)+sqrt.(halfDetTimeseries_fpeRev(parguessRes))))
                #plot!(0:899,(timeseries_ode(par_guess1Res)-sqrt.(halfDetTimeseries_fpeRev(parguessRes))))
                global MLE=[par_guess1Res[5],par_guess1Res[6],par_guess1Res[1],par_guess1Res[2],parguessRes[3],par_guess1Res[3],par_guess1Res[4],parguessRes[4],parguessRes[5],parguessRes[6]]
                if q<0.0
                    global LB2 = [0.0001, 0.0, q*1.05,-q*0.83/3000,0.0,omega_0-0.025,-10.0, (var_ww_0/500), 0.0, -20.0]
                    global UB2 = [0.04,1.0e-3,q*0.95,-q*1.2/300,1.0, omega_0+0.025,10.0, (var_ww_0*500), +120.0, 20.0]
                else
                    global LB2 = [0.0001, 0.0, q*0.95,-q*1.2/300,0.0,omega_0-0.025,-10.0, (var_ww_0/500), 0.0, -20.0]
                    global UB2 = [0.04,1.0e-3,q*1.05,-q*0.83/3000,1.0, omega_0+0.025,10.0,(var_ww_0*500), +120.0, +20.0]
                end
                global MLEres = optimize(loglike, LB2, UB2, MLE, Fminbox(LBFGS()),).minimizer
                #[posOfdbcrossing,MLE,RoCof,nadir,MeanDevInFirstHalf,Loglike]
                global info[h,:]=[turnOn,posOfdbcrossing,MLEres[1],MLEres[2],MLEres[3],MLEres[4],MLEres[5],MLEres[6],MLEres[7],MLEres[8],MLEres[9],MLEres[10],q,maximum(abs.(fs[(3600*(h-1)+1):(3600*(h-1)+900)])),maximum(abs.(smoothfs[1:450])),-loglike(MLEres)]
            end
        catch y
            global info[h,:]=[NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,maximum(abs.(fs[(3600*(h-1)+1):(3600*(h-1)+900)])),NaN,NaN]
        end
    else
        global info[h,:]=[NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]
    end
    println(h)
end
info[1,:]
h=1
plot(0:899,fs[((h-1)*3600+1):((h-1)*3600+900)])
plot!(0:899,timeseries_fpeNoC(info[1,5:12])[:,1])
#######################################
#######################################
#######################################


# if i find that the secnd does not match:
# move all the data such that it fits from nov on:
# instead of 00:00:00 we find 23:59:59 --> "overwrite one entry" by: global fs[(3600*(h-1)+1):end-1]=fs[(3600*(h-1)+2):end]
# instead of 00:00:00 we find 00:00:01 --> "delete one entry" by: global fs[(3600*(h-1)+1):end]=fs[(3600*(h-1)):end-1]   and same for the DateTime

h=1
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
for i in 861:900
    global smoothfs2[i]=sum(smoothfs[(900-(2*(900-i))):900])/(2*(900-i))
end
for i in 880:900
    global smoothfs2[i]=smoothfs2[879]
end
global fsbeg[1:30].=fs[((h-1)*3600)+1]
global fsbeg[31:90]=fs[((h-1)*3600+1):((h-1)*3600+60)]
for i in 1:30
    global i_smoothfs2[i]=sum(fsbeg[i:(i+60)])/61
    global i_smoothfs[i]=sum(fsbeg[26+i:34+i])/9
end
for i in 31:300
    global i_smoothfs[i]=sum(fs[(((h-1)*3600)+(i-4)):(((h-1)*3600)+(i+4))])/9
    global i_smoothfs2[i]=sum(fs[(((h-1)*3600)+(i-30)):(((h-1)*3600)+(i+30))])/61
end
for i in 11:290
    i_smoothfs3[i]=sum(i_smoothfs2[(i-10):(i+10)])/21
end
global i_smoothfs3[1:10]=i_smoothfs2[1:10]
global i_smoothfs3[291:300].=i_smoothfs3[290]
if polyfit(5:15,i_smoothfs3[5:15],1)[1]>0
    for i in 11:299
        if i_smoothfs3[i+1]<i_smoothfs3[i]
            global minpos2=i
            break
        end
    end
    for i in maximum([(minpos2-25),1]):minimum([(minpos2+25),300])
        if i_smoothfs[i+1]<i_smoothfs[i]
            global minpos=i
            break
        end
    end
else
    for i in 11:299
        if i_smoothfs3[i+1]>i_smoothfs3[i]
            global minpos2=i
            break
        end
    end
    for i in maximum([(minpos2-25),1]):minimum([(minpos2+25),300])
        if i_smoothfs[i+1]>i_smoothfs[i]
            global minpos=i
            break
        end
    end
end
if minpos>14
    global rocof=polyfit(1:minpos,i_smoothfs[1:minpos],1)[1]
    global turnOn=minimum([30,convert(Int64,abs(round((20/0.0045)*rocof)))])
end
plot(0:899,fs[((h-1)*3600+1):((h-1)*3600+900)])
plot!(0:299,i_smoothfs3)
plot!(0:299,i_smoothfs2)
plot!(0:299,i_smoothfs)
#plot!(0:899,smoothfs2)
#plot!(0:minpos-1,rocof(1:minpos))
global posOfdbcrossing=maximum([1,convert(Int64,round(minpos-(turnOn/2)))])
if turnOn<5
    global posOfdbcrossing = 0
elseif minpos<15
    global posOfdbcrossing = 0
elseif maximum(abs.(i_smoothfs[(minpos-10):(minpos+10)]))<0.13
    global posOfdbcrossing = 0
end
