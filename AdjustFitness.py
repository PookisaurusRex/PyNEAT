import numpy as np

def CalculateAdjustFitnesses_Wrong(self):
    # Filter out stagnated species, collect the set of non-stagnated species members, and compute their average adjusted fitness.
    # The average adjusted fitness scheme (normalized to the interval [0, 1]) allows the use of negative fitness values without
    # interfering with the shared fitness scheme. 
    for SpeciesID, Species in self.SpeciesSet.Species.items():
        Species.Fitness = self.Stagnation.SpeciesFitnessFunction(Species.GetFitnesses())

    AllFitnessValues = []
    RemainingSpecies = []
    for SpeciesID, Species in self.Stagnation.Update(self.SpeciesSet, self.Generation):
        if Species.bStagnant:
            self.Reporters.SpeciesStagnant(SpeciesID, Species)
        else:
            AllFitnessValues.extend(Genome.Fitness for Genome in Species.Members.values())
            RemainingSpecies.append(Species)
        
    if not RemainingSpecies: # No species left.
        self.SpeciesSet.Species = {}
        return 

    # GVand - Reimplement the original NEAT algorithm logic
    SumFitness = sum(AllFitnessValues)
    NumFitness = len(AllFitnessValues)
    for SingleSpecies in RemainingSpecies:
        GenomeAdjustedFitnesses = []
        for Genome in SingleSpecies.Members.values():
            SumOtherFitness = SumFitness - Genome.Fitness
            MeanOtherFitness = SumOtherFitness / (NumFitness-1)
            GenomeAdjustedFitnesses.append(Genome.Fitness - MeanOtherFitness)
        SingleSpecies.AdjustedFitness = sum(GenomeAdjustedFitnesses)
        if SingleSpecies.AdjustedFitness is None:
            SingleSpecies.AdjustedFitness = 0.0
    AdjustedFitnesses = [SingleSpecies.AdjustedFitness for SingleSpecies in RemainingSpecies]
    print('New Adj Fitness: {}'.format(AdjustedFitnesses))
    MinAdjFitness = min(AdjustedFitnesses)
    MaxAdjFitness = max(AdjustedFitnesses)

    if self.Reproduction.ReproductionConfig.AdjustedFitType == 'normalized':
        FitnessRange = max(self.Reproduction.ReproductionConfig.MinAdjustedFitnessRange, MaxAdjFitness - MinAdjFitness)
        for SingleSpecies in RemainingSpecies:
            AdjustedFitness = (SingleSpecies.AdjustedFitness-MinAdjFitness) / FitnessRange
            SingleSpecies.AdjustedFitness = AdjustedFitness
            if SingleSpecies.AdjustedFitness is None:
                SingleSpecies.AdjustedFitness = 0.0
        AdjustedFitnesses = [SingleSpecies.AdjustedFitness for SingleSpecies in RemainingSpecies]
        for SingleSpecies, AdjustedFitness in zip(RemainingSpecies, AdjustedFitnesses):
            SingleSpecies.AdjustedFitness = AdjustedFitness / len(SingleSpecies.Members.values())
        #print('Normalized Adj Fitness: {}'.format(AdjustedFitnesses))
    elif self.Reproduction.ReproductionConfig.AdjustedFitType == 'linear':
        if MinAdjFitness < 0.0:
            AdjustedFitnesses = [AdjustedFitness + abs(MinAdjFitness) for AdjustedFitness in AdjustedFitnesses]
        for SingleSpecies, AdjustedFitness in zip(RemainingSpecies, AdjustedFitnesses):
            SingleSpecies.AdjustedFitness = AdjustedFitness / len(SingleSpecies.Members.values())
        pass


def CalculateAdjustFitnesses(self):
    # Filter out stagnated species, collect the set of non-stagnated species members, and compute their average adjusted fitness.
    # The average adjusted fitness scheme (normalized to the interval [0, 1]) allows the use of negative fitness values without
    # interfering with the shared fitness scheme. 
    AllFitnessValues = []
    RemainingSpecies = []
    for SpeciesID, Species in self.Stagnation.Update(self.SpeciesSet, self.Generation):
        if Species.bStagnant:
            self.Reporters.SpeciesStagnant(SpeciesID, Species)
        else:
            AllFitnessValues.extend(Genome.Fitness for Genome in Species.Members.values())
            RemainingSpecies.append(Species)
    if not RemainingSpecies: # No species left.
        self.SpeciesSet.Species = {}
        return 
    # GVand - Reimplement the original NEAT algorithm logic
    AdjustedFitnesses = []
    for SingleSpecies in RemainingSpecies:
        NumMembersOfSpecies = len(SingleSpecies.Members.values())
        GenomeAdjustedFitnesses = [(Genome.Fitness / NumMembersOfSpecies) for Genome in SingleSpecies.Members.values()]
        AdjustedFitnesses.append(np.mean(GenomeAdjustedFitnesses))
    #print('New Adj Fitness: {}'.format(AdjustedFitnesses))
    MinAdjFitness = min(AdjustedFitnesses)
    MaxAdjFitness = max(AdjustedFitnesses)
    FitnessRange = max(self.Reproduction.ReproductionConfig.MinAdjustedFitnessRange, MaxAdjFitness - MinAdjFitness)
    for SingleSpecies, AdjustedFitness in zip(RemainingSpecies, AdjustedFitnesses):
        SingleSpecies.AdjustedFitness = (AdjustedFitness - MinAdjFitness) / len(SingleSpecies.Members.values())
        if self.Reproduction.ReproductionConfig.AdjustedFitType == 'normalized':
            SingleSpecies.AdjustedFitness /= FitnessRange
        if SingleSpecies.AdjustedFitness is None:
            raise RuntimeError("Fitness not assigned to species {}".format(SingleSpecies.Key))