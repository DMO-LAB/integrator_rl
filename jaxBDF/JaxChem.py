import numpy as np
from jax import grad, jit
from jax import lax
from jax import random
import jax
import jax.numpy as jnp
from ReadChemData import ParseChemFile
import itertools
from jax import config
import time
import sys
#config.update("jax_enable_x64", True)
#jax.config.update('jax_platform_name', 'cpu')

class JaxChem():
    def __init__(self, file_name):
        chem = ParseChemFile(filename = file_name)
        chem.get_reaction_parameters()
        chem.get_thermo_data()

        self.reversible_reaction = np.asarray(chem.reversible_reaction)
        self.species_names = chem.species_names
        self.elements = chem.elements
        self.element_weights = chem.element_weights
        self.eqn_type_id = chem.eqn_type_id
        self.num_species = chem.num_species
        self.num_reactions = chem.num_reactions
        self.species_composition = chem.species_compositon
        self.molecular_weights = jnp.asarray(np.array(list(chem.molecular_weights.values()))[None,:])

        self.read_default_reactions_data(chem) # read reaction parameter for default reactions
        self.read_3body_reactions_data(chem) # read reaction parameter for three body reactions
        self.read_falloff_reactions_data(chem) # read reaction parameter for fall-off reactions
        self.read_troefalloff_reactions_data(chem)
        self.read_thermo_coefficients(chem) # read a0 to a6 for computing thermodynamic properties

        self.nu = jnp.vstack([self.nu_default, self.nu_3body, self.nu_falloff, self.nu_troefalloff])
        self.reversible_inds = jnp.hstack([self.reversible_inds_default, self.reversible_inds_3body, self.reversible_inds_falloff, self.reversible_inds_troefalloff])
      
    def species_index(self, species_name):
        try:
            index_ = self.species_names.index(species_name)
        except SystemExit:
            print("error:" + species_name + "is not among the list of species in the mechanism")
        return index_
    
    def num_atoms(self, species_name, element_name):
        return self.species_composition[species_name].get(element_name, 0)
    
    def init_from_equi_ratio(self, oxidizer, fuel, phi):
        fuel_species = list(fuel)
        atoms_fuel = {}
        for el in self.elements:
            num_atoms = 0
            for sp in fuel_species:
                num_atoms += self.num_atoms(sp, el)*fuel[sp]
            atoms_fuel[el] = num_atoms
        
        atoms_ox = {}
        ox_species = list(oxidizer)
        for el in self.elements:
            num_atoms = 0
            for sp in ox_species:
                num_atoms += self.num_atoms(sp, el)*oxidizer[sp]
            atoms_ox[el] = num_atoms
            
        A = np.array([[atoms_ox["O"], -1, -2], [atoms_ox["C"], 0, -1], [atoms_ox["H"], -2, 0]])
        b = np.array([[-atoms_fuel["O"]],[-atoms_fuel["C"]],[-atoms_fuel["H"]]])
        x = np.matmul(np.linalg.inv(A), b)
        no_moles_ox_st = x[0][0]
        no_moles_ox = no_moles_ox_st/phi

        MW_1D = self.molecular_weights.reshape(-1)
        MW_ox = np.sum([MW_1D[self.species_index(sp)]*oxidizer[sp] for sp in ox_species])
        MW_fuel = np.sum([MW_1D[self.species_index(sp)]*fuel[sp] for sp in fuel_species])
        
        tot_mass = MW_fuel + MW_ox*no_moles_ox

        Y_fuel = MW_fuel/tot_mass
        Y_ox = MW_ox*no_moles_ox/tot_mass

        Y = jnp.zeros(shape = (self.num_species,))
        for sp in ox_species:
            isp = self.species_index(sp)
            Y_sp = oxidizer[sp]*MW_1D[isp]/MW_ox
            Y = Y.at[isp].set(Y_ox*Y_sp)
        
        for sp in fuel_species:
            isp = self.species_index(sp)
            Y_sp = fuel[sp]*MW_1D[isp]/MW_fuel
            Y = Y.at[isp].set(Y_fuel*Y_sp)

        #self.stoichimetic_AF = MW_fuel/(MW_ox*no_moles_ox)

        #no_moles_ox = no_moles_ox
        return Y #self.stoichimetic_AF
    
    def initialize_from_equiv_ratio(self, phi):
        self.phi = phi


    def read_default_reactions_data(self, chem):
        inds_default = self.eqn_type_id['default']
        self.nu_reactants_default = jnp.asarray(chem.nu_reactant[inds_default])
        self.nu_products_default = jnp.asarray(chem.nu_product[inds_default])
        self.nu_default = self.nu_products_default - self.nu_reactants_default
        self.A_default = jnp.asarray(chem.A[inds_default])
        self.b_default = jnp.asarray(chem.b[inds_default])
        self.E_default = jnp.asarray(chem.E[inds_default])

        self.reversible_inds_default = jnp.asarray(self.reversible_reaction[inds_default])

    def read_3body_reactions_data(self, chem):
        inds_3body = self.eqn_type_id['three-body']
        self.nu_reactants_3body = jnp.asarray(chem.nu_reactant[inds_3body])
        self.nu_products_3body = jnp.asarray(chem.nu_product[inds_3body])
        self.nu_3body = self.nu_products_3body - self.nu_reactants_3body
        self.A_3body = jnp.asarray(chem.A[inds_3body])
        self.b_3body = jnp.asarray(chem.b[inds_3body])
        self.E_3body = jnp.asarray(chem.E[inds_3body])
        self.efficiencies_3body = jnp.asarray(chem.efficiencies[inds_3body])

        self.reversible_inds_3body = jnp.asarray(self.reversible_reaction[inds_3body])

    def read_falloff_reactions_data(self, chem):
        inds_falloff = self.eqn_type_id['falloff']
        self.nu_reactants_falloff = jnp.asarray(chem.nu_reactant[inds_falloff])
        self.nu_products_falloff = jnp.asarray(chem.nu_product[inds_falloff])
        self.nu_falloff = self.nu_products_falloff - self.nu_reactants_falloff
        self.highA_falloff = jnp.asarray(chem.highA[inds_falloff])
        self.highb_falloff = jnp.asarray(chem.highb[inds_falloff])
        self.highE_falloff = jnp.asarray(chem.highE[inds_falloff])
        self.lowA_falloff = jnp.asarray(chem.lowA[inds_falloff])
        self.lowb_falloff = jnp.asarray(chem.lowb[inds_falloff])
        self.lowE_falloff = jnp.asarray(chem.lowE[inds_falloff])
        self.efficiencies_falloff = jnp.asarray(chem.efficiencies[inds_falloff])

        self.reversible_inds_falloff = jnp.asarray(self.reversible_reaction[inds_falloff])

    def read_troefalloff_reactions_data(self, chem):
        inds_troefalloff = self.eqn_type_id['troe-falloff']
        self.nu_reactants_troefalloff = jnp.asarray(chem.nu_reactant[inds_troefalloff])
        self.nu_products_troefalloff = jnp.asarray(chem.nu_product[inds_troefalloff])
        self.nu_troefalloff = self.nu_products_troefalloff - self.nu_reactants_troefalloff
        self.highA_troefalloff = jnp.asarray(chem.highA[inds_troefalloff])
        self.highb_troefalloff = jnp.asarray(chem.highb[inds_troefalloff])
        self.highE_troefalloff = jnp.asarray(chem.highE[inds_troefalloff])
        self.lowA_troefalloff = jnp.asarray(chem.lowA[inds_troefalloff])
        self.lowb_troefalloff = jnp.asarray(chem.lowb[inds_troefalloff])
        self.lowE_troefalloff = jnp.asarray(chem.lowE[inds_troefalloff])
        self.troeA = jnp.asarray(chem.TroeA[inds_troefalloff])
        self.troeT1 = jnp.asarray(chem.TroeT1[inds_troefalloff])
        self.troeT3 = jnp.asarray(chem.TroeT3[inds_troefalloff])
        self.troeT2 = jnp.asarray(chem.TroeT2[inds_troefalloff])
        self.efficiencies_troefalloff = jnp.asarray(chem.efficiencies[inds_troefalloff])

        self.reversible_inds_troefalloff = jnp.asarray(self.reversible_reaction[inds_troefalloff])

    def read_thermo_coefficients(self, chem):
        print(chem.thermo_coefficients)
        A1_np = np.array([chem.thermo_coefficients[sp_name][0] for sp_name in self.species_names])
        A2_np = np.array([chem.thermo_coefficients[sp_name][1] for sp_name in self.species_names])
        self.A1 = jnp.asarray(A1_np)
        self.A2 = jnp.asarray(A2_np)

    def compute_net_production_rates(self, alpha):


        C3D = self.C.T[:,:,None]

        self.nu_reactants = jnp.vstack([self.nu_reactants_default, self.nu_reactants_3body, self.nu_reactants_falloff, self.nu_reactants_troefalloff])
        self.nu_products = jnp.vstack([self.nu_products_default, self.nu_products_3body, self.nu_products_falloff, self.nu_products_troefalloff])
        forward_progress = jnp.prod(C3D**self.nu_reactants.T, axis = 1) * self.kf.T
        backward_progress = jnp.prod(C3D**self.nu_products.T, axis = 1) * self.kr.T

        #forward_progress = jnp.exp(jnp.matmul(jnp.log(self.C.T + 1e-10), self.nu_reactants.T)) * self.kf.T
        #backward_progress = jnp.exp(jnp.matmul(jnp.log(self.C.T + 1e-10), self.nu_products.T)) * self.kr.T

        
        sig = 1 - jnp.clip(jnp.abs(self.nu), a_max = 1)

        alpha_reac = jnp.min(alpha+sig, axis = 1)


        self.net_progress = (forward_progress - backward_progress)*alpha_reac[None,:]
        
        self.net_production_rates = self.net_progress @ self.nu

    def compute_net_production_ratesf(self, alpha):


        C3D = self.C.T[:,:,None]

        self.nu_reactants = jnp.vstack([self.nu_reactants_default, self.nu_reactants_3body, self.nu_reactants_falloff, self.nu_reactants_troefalloff])
        
        #self.nu_products = jnp.vstack([self.nu_products_default, self.nu_products_3body, self.nu_products_falloff, self.nu_products_troefalloff])
        
        forward_progress = jnp.prod(C3D**self.nu_reactants.T, axis = 1) * self.kf.T
        
        #backward_progress = jnp.prod(C3D**self.nu_products.T, axis = 1) * self.kr.T

        sig = 1 - jnp.clip(jnp.abs(self.nu), a_max = 1)

        alpha_reac = jnp.min(alpha+sig, axis = 1)


        self.net_progress = (forward_progress)*alpha_reac[None,:]
        
        self.net_production_rates = self.net_progress @ self.nu_reactants

    def update_thermo_properties(self):
        self.Y = self.Y/jnp.sum(self.Y, axis = 0, keepdims=True)

        self.Tinv = 1/self.T; # inverse of temperature
        self.T2 = self.T*self.T
        self.T3 = self.T2*self.T
        self.T4 = self.T2*self.T2
        self.logT = jnp.log(self.T)

        self.Rc = 1.98720425864083; # universal gas constant in cal/mol-K
        self.Ru = 8.31446261815324e7; # universal gas constant in ergs/mol-K (cgs)
        
        self.one_atm_in_cgs = 1.01325e+6 # one atmosphere of pressure in cgs units
        self.Pcgs =  self.Patm * self.one_atm_in_cgs # P in dynes/cm2
        

        self.mix_molecular_weight = 1/jnp.matmul(1/self.molecular_weights, self.Y)
        self.Rgas = self.Ru/self.mix_molecular_weight; #specific gas constant in ergs/g-K
        self.X = self.Y * self.mix_molecular_weight/self.molecular_weights.T
        self.density = self.Pcgs/(self.Rgas * self.T)
        self.C = self.X * self.Pcgs/(self.Ru * self.T)
        
    
    def compute_falloff_kf(self):
        
        k0 = self.lowA_falloff * self.T ** self.lowb_falloff * jnp.exp(-self.lowE_falloff/(self.Rc * self.T))

        kinf = self.highA_falloff * self.T ** self.highb_falloff * jnp.exp(-self.highE_falloff/(self.Rc * self.T))

        conc_M_falloff = jnp.matmul(self.efficiencies_falloff, self.C)

        Pr = k0 * conc_M_falloff/kinf
        
        kf_falloff = kinf * (Pr / (1 + Pr))

        return kf_falloff

    def compute_troefalloff_kf(self):
        
        k0 = self.lowA_troefalloff * self.T ** self.lowb_troefalloff * jnp.exp(-self.lowE_troefalloff/(self.Rc * self.T))

        kinf = self.highA_troefalloff * self.T ** self.highb_troefalloff * jnp.exp(-self.highE_troefalloff/(self.Rc * self.T))

        conc_M_troefalloff = jnp.matmul(self.efficiencies_troefalloff, self.C)

        Pr = k0 * conc_M_troefalloff/kinf
        
        Fcent = (1 - self.troeA) * jnp.exp(-self.T/self.troeT3) + self.troeA*jnp.exp(-self.T/self.troeT1) + jnp.exp(-self.troeT2 * self.Tinv)

        logFcent = jnp.log10(Fcent)
        logPr = jnp.log10(Pr)
        c = -0.4 - 0.67 * logFcent
        n = 0.75 - 1.27 * logFcent
        numer = logPr + c
        denum = n - 0.14 * (logPr + c)
        logF = 1/(1 + (numer/denum)**2) * logFcent
        F = 10**(logF)
        kf_troefalloff = kinf * (Pr / (1 + Pr)) * F

        return kf_troefalloff
    
    def compute_forward_rate_constants(self):
        self.kf_default = self.A_default * self.T ** self.b_default * jnp.exp(-self.E_default/(self.Rc * self.T))

        conc_M_3body = jnp.matmul(self.efficiencies_3body, self.C)
        self.kf_3body = conc_M_3body * self.A_3body * self.T ** self.b_3body * jnp.exp(-self.E_3body/(self.Rc * self.T))

        self.kf_falloff = self.compute_falloff_kf()

        self.kf_troefalloff = self.compute_troefalloff_kf()

        self.kf = jnp.vstack([self.kf_default, self.kf_3body, self.kf_falloff, self.kf_troefalloff])

        return
    
    def compute_reverse_rate_constants(self):

        #t1 = time.time()


        self.cp_R1 = self.A1[:,0:1] + self.A1[:,1:2]*self.T + self.A1[:,2:3] * self.T2 + self.A1[:,3:4] * self.T3 + self.A1[:,4:5] * self.T4 # 
        self.cp_R2 = self.A2[:,0:1] + self.A2[:,1:2]*self.T + self.A2[:,2:3] * self.T2 + self.A2[:,3:4] * self.T3 + self.A2[:,4:5] * self.T4 # 

        self.h_RT1 = self.A1[:,0:1] + self.A1[:,1:2]/2*self.T + self.A1[:,2:3]/3 * self.T2 + self.A1[:,3:4]/4 * self.T3 + self.A1[:,4:5]/5 * self.T4 + self.A1[:,5:6] * self.Tinv
        self.h_RT2 = self.A2[:,0:1] + self.A2[:,1:2]/2*self.T + self.A2[:,2:3]/3 * self.T2 + self.A2[:,3:4]/4 * self.T3 + self.A2[:,4:5]/5 * self.T4 + self.A2[:,5:6] * self.Tinv

        self.s_R1 = self.A1[:,0:1]*self.logT + self.A1[:,1:2]*self.T + 0.5 * self.A1[:,2:3] * self.T2 + self.A1[:,3:4]/3 * self.T3 + 0.25 * self.A1[:,4:5] * self.T4 + self.A1[:,6:7]
        self.s_R2 = self.A2[:,0:1]*self.logT + self.A2[:,1:2]*self.T + 0.5 * self.A2[:,2:3] * self.T2 + self.A2[:,3:4]/3 * self.T3 + 0.25 * self.A2[:,4:5] * self.T4 + self.A2[:,6:7]

        #print(jnp.where(T.T > 500)
        #print(jnp.tile(T.T > 1000, reps = 9).T)

        self.cond = jnp.tile(self.T.T < 1000, reps = self.num_species).T
        self.cp_R = jax.lax.select(self.cond, self.cp_R1, self.cp_R2)
        
        self.h_RT = jax.lax.select(self.cond, self.h_RT1, self.h_RT2)
        self.s_R = jax.lax.select(self.cond, self.s_R1, self.s_R2)
        #print((self.cp_R).shape)

        self.s1 = self.s_R - self.h_RT
        
        delta_S_R = jnp.matmul(self.nu, self.s_R)
        delta_H_RT = jnp.matmul(self.nu, self.h_RT)

        Kp = jnp.exp(delta_S_R - delta_H_RT)
        
        Kc = Kp * (self.one_atm_in_cgs/(self.Ru*self.T)) ** jnp.sum(self.nu, axis = 1, keepdims = True)

        self.kr = self.kf/Kc*self.reversible_inds[:,None]

        return
        #print("kr time = ", t7 - t6) 

    def compute_source_terms(self, state):
        alpha = jnp.ones(shape = (184,))
        
        self.T = state[:,0:1].T
        self.Patm = state[:,1:2].T
        self.Y = state[:,2:].T
        
        #t1 = time.time()
        self.update_thermo_properties()
        #t2 = time.time()
        #print("update properties time = ", t2-t1)
        self.compute_forward_rate_constants()
        #t3 = time.time()
        #print("forward rates time = ", t3-t2)
        self.compute_reverse_rate_constants()
        #t4 = time.time()
        #print("reverse rates time = ", t4-t3)
        self.compute_net_production_rates(alpha)
        #t5 = time.time()
        #print("net production rates time = ", t5-t4)

        self.species_rates = (self.net_production_rates * self.molecular_weights).T/self.density
        partial_molar_cp = self.cp_R * self.Ru
        mixture_cp_mole = jnp.sum(partial_molar_cp * self.X, axis = 0, keepdims=True)
        mixture_cp_mass = mixture_cp_mole/self.mix_molecular_weight
        partial_molar_enthalpies = self.h_RT * self.Ru * self.T
        partial_mass_enthalpies = partial_molar_enthalpies/self.molecular_weights.T
        
        self.temperature_source = -jnp.sum(self.species_rates * partial_mass_enthalpies/mixture_cp_mass, axis = 0)

        #t6 = time.time()
        #print("source term rates time = ", t6-t5)
        return jnp.concatenate((self.temperature_source[:,None], self.species_rates.T), axis = 1)
        #return jnp.mean(jnp.concatenate((self.temperature_source[:,None], self.species_rates.T), axis = 1))/1e15 #self.temperature_source, self.species_rates

if __name__ == '__main__':
    config.update("jax_enable_x64", True)
    #jax.config.update('jax_platform_name', 'cpu')

    num_points = 1

    key = jax.random.PRNGKey(np.random.randint(10000))

    jC = JaxChem(file_name = "ch4_30species.yaml")

    #jC = JaxChem(file_name = "chem_ch4.yaml")

    #Y = jC.init_from_equi_ratio(oxidizer = {'O2': 1.0, 'N2': 3.76}, fuel = {"CH4":1.0}, phi = 0.7)[:,None]

    Y = random.uniform(key, shape=(jC.num_species, num_points), minval=0.0, maxval=1.0)
    Y = Y/jnp.sum(Y, axis = 0, keepdims=True)
    #print(Y)
    Patm = random.uniform(key, shape=(1, num_points), minval=1., maxval=40.); # in atmosphere
    T = random.uniform(key, shape=(1, num_points), minval=300., maxval=2000.); # in Kelvin

    state = jnp.concatenate((T.T, Patm.T, Y.T), axis = 1)

    source_jit = jax.jit(jC.compute_source_terms)

    source_terms = source_jit(state).block_until_ready()
    
    import time
    t1 = time.time()

    source_terms = source_jit(state).block_until_ready()

    print(time.time() - t1)

    #jC.compute_source_terms(state[1:2,:]).block_until_ready()

    source_terms = jC.compute_source_terms(state).block_until_ready()

    #print(source_terms)

    jnp.save("states", state)
    jnp.save("source_terms", source_terms)
    #X = jnp.zeros(shape = (jC.num_reactions,1))
    #inds = jnp.array(jC.eqn_type_id['default'])
    #X.at[inds].set(jC.kf_default)

    
    #C3D = jC.C.T[:,:,None]
    #jnp.prod(C3D**jC.nu_reactants_default.T, axis = 1)
    from jax import jacfwd, jacrev
    jac_jitted = jax.jit(jacfwd(jC.compute_source_terms))
    J = jac_jitted(state).block_until_ready()
    t = time.time()
    print("time elaspsed for Jac is", time.time() - t, "seconds")
    #inds = jnp.where(jnp.prod(C3D**jC.nu_reactants_default.T, axis = 1)[0] != 0.0)[0]
    #jnp.array(jC.eqn_type_id['default'])[inds] + 1

# import yaml
# from yaml.resolver import Resolver

# for ch in "OoYyNn":
#     if len(Resolver.yaml_implicit_resolvers[ch]) == 1:
#         del Resolver.yaml_implicit_resolvers[ch]
#     else:
#         Resolver.yaml_implicit_resolvers[ch] = [x for x in
#                 Resolver.yaml_implicit_resolvers[ch] if x[0] != 'tag:yaml.org,2002:bool']
        
# with open("chem_ch4.yaml", "r") as stream:
#     try:
#         data = yaml.full_load(stream)
#     except yaml.YAMLError as exc:
#         print(exc)