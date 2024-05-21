# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:54:23 2023

@author: oowoyele
"""

import numpy as np
import yaml
import sys

from yaml.resolver import Resolver

for ch in "OoYyNn":
    if len(Resolver.yaml_implicit_resolvers[ch]) == 1:
        del Resolver.yaml_implicit_resolvers[ch]
    else:
        Resolver.yaml_implicit_resolvers[ch] = [x for x in
                Resolver.yaml_implicit_resolvers[ch] if x[0] != 'tag:yaml.org,2002:bool']
        
def strip_(string, char_list):
    for char in char_list:
        string = string.strip(char)
    
    return string.strip()

class ParseChemFile():
    def __init__(self, filename):
        
        with open(filename, "r") as stream:
            try:
                self.data = yaml.full_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        
        self.large_number = 1e6
        self.species_names, self.elements = self.get_species_names()

        self.reactions = self.data['reactions']
        
        self.num_species = len(self.species_names)
        self.num_reactions = len(self.reactions)
        self.nu_reactant = np.zeros(shape = (self.num_reactions, self.num_species))
        self.nu_product = np.zeros(shape = (self.num_reactions, self.num_species))
        
        self.eqn_type = [[]]*self.num_reactions
        
        self.TroeA = np.zeros(shape = (self.num_reactions,1))
        self.TroeT3 = np.zeros(shape = (self.num_reactions,1))
        self.TroeT1 = np.zeros(shape = (self.num_reactions,1))
        self.TroeT2 = np.zeros(shape = (self.num_reactions,1)) + self.large_number # give high value so term goes to 0.

        self.lowA = np.zeros(shape = (self.num_reactions,1))
        self.lowb = np.zeros(shape = (self.num_reactions,1))
        self.lowE = np.zeros(shape = (self.num_reactions,1))
        
        self.highA = np.zeros(shape = (self.num_reactions,1))
        self.highb = np.zeros(shape = (self.num_reactions,1))
        self.highE = np.zeros(shape = (self.num_reactions,1))
        
        self.A = np.zeros(shape = (self.num_reactions,1))
        self.b = np.zeros(shape = (self.num_reactions,1))
        self.E = np.zeros(shape = (self.num_reactions,1))
        
        self.efficiencies = np.ones(shape = (self.num_reactions, self.num_species))
        
        self.reversible_reaction = []

        self.eqn_type_id = {'default': [], 'troe-falloff': [], 'falloff': [], 'three-body': []}
    
    def get_species_names(self):
        phase_data = dict(self.data['phases'][0])
        #species_names = phase_data['species']
        return phase_data['species'], phase_data['elements']
    
    def get_reaction_parameters(self):
        for ireact, reaction in enumerate(self.reactions):
            reaction_dict = dict(reaction)

            #print(ireact, reaction_dict['equation'])

            reactants_and_products = reaction_dict['equation']
            #print(reactants_and_products)
            if "<=>" in reactants_and_products:
                reactants_products_split = reactants_and_products.split("<=>")
                self.reversible_reaction += [1]
            elif " = " in reactants_and_products:
                reactants_products_split = reactants_and_products.split(" = ")
                self.reversible_reaction += [1]
            elif "=>" in reactants_and_products:
                reactants_products_split = reactants_and_products.split("=>")
                self.reversible_reaction += [0]
            else:
                print(reactants_and_products)
                sys.exit("chemical equation is incomplete. Missing =>, =, or <=>")

            reactants_products_split = [reacprod.strip() for reacprod in reactants_products_split]

            reactants_products_split = [reacprod.replace("(+M)","") for reacprod in reactants_products_split]
            reactants_products_split = [reacprod.replace("(+ M)","") for reacprod in reactants_products_split]

            reactants = reactants_products_split[0].split(" + ")
            reactants_striped = [reactant.strip() for reactant in reactants]
            
            products = reactants_products_split[1].split(" + ")
            products_striped = [product.strip() for product in products]
            
            #print(reactants)
            #reactants_striped = [strip_(reactant, [None]) for reactant in reactants]
            #print(reactants_striped)
            #products_striped = [strip_(product, [None]) for product in products]

            reactants_list = [reactant.split() for reactant in reactants_striped]
            products_list = [product.split() for product in products_striped]
            
            
            #print(reactants)
            for reactant in reactants_list:
                if len(reactant) == 2:
                    #print(reactant)
                    #asc=saccsa
                    self.nu_reactant[ireact, self.species_names.index(reactant[1])] += int(reactant[0])
                elif len(reactant) == 1:
                    try:
                        self.nu_reactant[ireact, self.species_names.index(reactant[0])] += 1.0
                    except:
                        None
                        
            for product in products_list:
                if len(product) == 2:
                    self.nu_product[ireact, self.species_names.index(product[1])] += int(product[0])
                elif len(product) == 1:
                    try:
                        self.nu_product[ireact, self.species_names.index(product[0])] += 1.0
                    except:
                        None

            if 'type' not in reaction_dict.keys():
                self.eqn_type_id['default'] += [ireact]
                self.eqn_type[ireact] = 'default'
            else:
                self.eqn_type[ireact] = reaction_dict['type']
            
            if self.eqn_type[ireact] == 'default':
                self.A[ireact], self.b[ireact], self.E[ireact] = reaction_dict['rate-constant'].values()
            elif self.eqn_type[ireact] == 'three-body':
                self.eqn_type_id['three-body'] += [ireact]
                self.A[ireact], self.b[ireact], self.E[ireact] = reaction_dict['rate-constant'].values()
                efficiencies_dict = reaction_dict['efficiencies']
                species_in_dict = efficiencies_dict.keys()
                for spec in species_in_dict:
                    self.efficiencies[ireact, self.species_names.index(spec)] = efficiencies_dict[spec]
                continue #sys.exit()
            elif self.eqn_type[ireact] == 'falloff':
                if 'Troe' in reaction_dict.keys():                
                    self.eqn_type_id['troe-falloff'] += [ireact]
                    self.TroeA[ireact] = reaction_dict['Troe']['A']
                    self.TroeT1[ireact] = reaction_dict['Troe']['T1']
                    self.TroeT3[ireact] = reaction_dict['Troe']['T3']
                    try:
                        self.TroeT2[ireact] = reaction_dict['Troe']['T2']
                    except:
                        print("T2 not specified for reaction ", str(ireact))

                else:
                    self.eqn_type_id['falloff'] += [ireact]

                self.lowA[ireact], self.lowb[ireact], self.lowE[ireact] = reaction_dict['low-P-rate-constant'].values()
                self.highA[ireact], self.highb[ireact], self.highE[ireact] = reaction_dict['high-P-rate-constant'].values()

                #print(ireact)
                
                #self.TroeA[ireact], self.TroeT3[ireact], self.TroeT1[ireact] = reaction_dict['Troe'].values() # come back here later. In some cases, we have T2

                efficiencies_dict = reaction_dict['efficiencies']
                species_in_dict = efficiencies_dict.keys()
                for spec in species_in_dict:
                    self.efficiencies[ireact, self.species_names.index(spec)] = efficiencies_dict[spec]
                
    def get_thermo_data(self):
        #element_weights = {'H': 1.00797, 'O': 15.9994, 'N': 14.0067, 'Ar': 39.948}
        self.element_weights = {'C': 12.011, 'H': 1.008, 'O': 15.999, 'N': 14.007, 'Ar': 39.948}
        species_data = self.data['species']
        self.molecular_weights = {}
        self.thermo_coefficients = {}
        self.species_compositon = {}
        for sp_data in species_data:
            name = sp_data['name']
            composition = sp_data['composition']
            self.species_compositon[name] = composition
            keys = composition.keys()
            self.molecular_weights[name] = 0.
            self.thermo_coefficients[name] = sp_data['thermo']['data']
            for key in keys:
                self.molecular_weights[name] += composition[key] * self.element_weights[key]