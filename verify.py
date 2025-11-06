import numpy as np

class DSTheory:
    """Classe pour implémenter la théorie de Dempster-Shafer et la méthode proposée"""
    
    def __init__(self, frame_of_discernment):
        """
        Initialisation
        frame_of_discernment: liste des hypothèses, ex: ['A', 'B', 'C']
        """
        self.frame = frame_of_discernment
        self.n = len(frame_of_discernment)
    
    def pignistic_transform(self, mass):
        """
        Transforme une masse en probabilité pignistique
        mass: dict avec les hypothèses comme clés
        Returns: vecteur numpy de dimension n
        """
        pignistic = np.zeros(self.n)
        
        for hypothesis, m_value in mass.items():
            if hypothesis == 'theta' or m_value == 0:
                continue
                
            # Séparer les hypothèses composites (ex: 'AC' -> ['A', 'C'])
            elements = list(hypothesis)
            card = len(elements)
            
            if card > 0:
                for elem in elements:
                    if elem in self.frame:
                        idx = self.frame.index(elem)
                        pignistic[idx] += m_value / card
        
        return pignistic
    
    def cosine_similarity(self, mass1, mass2):
        """
        Calcule la similarité cosinus entre deux masses
        """
        pig1 = self.pignistic_transform(mass1)
        pig2 = self.pignistic_transform(mass2)
        
        dot_product = np.dot(pig1, pig2)
        norm1 = np.linalg.norm(pig1)
        norm2 = np.linalg.norm(pig2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def conflict_degree(self, mass1, mass2):
        """
        Calcule le degré de conflit: ConfDegree = 1 - cos(m1, m2)
        """
        return 1 - self.cosine_similarity(mass1, mass2)
    
    def build_correlation_matrix(self, masses):
        """
        Construit la matrice de corrélation S
        masses: liste de dicts de masses
        """
        k = len(masses)
        S = np.ones((k, k))
        
        for i in range(k):
            for j in range(k):
                if i != j:
                    S[i][j] = self.cosine_similarity(masses[i], masses[j])
        
        return S
    
    def calculate_credibility(self, masses):
        """
        Calcule le vecteur de crédibilité Crd
        """
        S = self.build_correlation_matrix(masses)
        k = len(masses)
        
        # Calcul du vecteur de support Sup
        Sup = np.sum(S, axis=1)  # Somme sur chaque ligne
        
        # Normalisation
        Sum_Sup = np.sum(Sup)
        Crd = Sup / Sum_Sup if Sum_Sup > 0 else np.ones(k) / k
        
        return Crd
    
    def weighted_average_mass(self, masses, credibility):
        """
        Calcule la moyenne pondérée des masses MAE(m)
        """
        weighted_mass = {}
        
        # Obtenir toutes les hypothèses
        all_hypotheses = set()
        for mass in masses:
            all_hypotheses.update(mass.keys())
        
        # Calculer la moyenne pondérée
        for hyp in all_hypotheses:
            weighted_mass[hyp] = sum(
                credibility[i] * masses[i].get(hyp, 0) 
                for i in range(len(masses))
            )
        
        return weighted_mass
    
    def dempster_combination(self, mass1, mass2):
        """
        Règle de combinaison de Dempster
        """
        combined = {}
        K = 0  # Conflit
        
        for h1, m1 in mass1.items():
            for h2, m2 in mass2.items():
                # Intersection des hypothèses
                intersection = self._intersection(h1, h2)
                
                if intersection == '':
                    K += m1 * m2
                else:
                    combined[intersection] = combined.get(intersection, 0) + m1 * m2
        
        # Normalisation
        if K < 1:
            for key in combined:
                combined[key] /= (1 - K)
        
        return combined, K
    
    def _intersection(self, h1, h2):
        """Helper pour calculer l'intersection de deux hypothèses"""
        if h1 == 'theta' or h2 == 'theta':
            return h2 if h1 == 'theta' else h1
        
        set1 = set(h1)
        set2 = set(h2)
        inter = set1.intersection(set2)
        
        return ''.join(sorted(inter))
    
    def murphy_combination(self, masses):
        """
        Règle de combinaison de Murphy (moyenne simple puis Dempster)
        """
        # Moyenne simple
        avg_mass = {}
        all_hypotheses = set()
        for mass in masses:
            all_hypotheses.update(mass.keys())
        
        for hyp in all_hypotheses:
            avg_mass[hyp] = sum(mass.get(hyp, 0) for mass in masses) / len(masses)
        
        # Application de Dempster k fois
        result = avg_mass
        for _ in range(len(masses) - 1):
            result, _ = self.dempster_combination(result, avg_mass)
        
        return result
    
    def yager_combination(self, mass1, mass2):
        """
        Règle de combinaison de Yager
        """
        combined = {}
        conflict_mass = 0
        
        for h1, m1 in mass1.items():
            for h2, m2 in mass2.items():
                intersection = self._intersection(h1, h2)
                
                if intersection == '':
                    conflict_mass += m1 * m2
                else:
                    combined[intersection] = combined.get(intersection, 0) + m1 * m2
        
        # Le conflit est assigné à theta (cadre de discernement)
        combined['theta'] = combined.get('theta', 0) + conflict_mass
        
        return combined
    
    def proposed_method(self, masses):
        """
        Méthode proposée dans le papier
        """
        # Étape 1: Calculer la crédibilité
        credibility = self.calculate_credibility(masses)
        
        # Étape 2: Moyenne pondérée
        weighted_mass = self.weighted_average_mass(masses, credibility)
        
        # Étape 3: Appliquer Murphy sur la masse pondérée
        result = weighted_mass
        for _ in range(len(masses) - 1):
            result, _ = self.dempster_combination(result, weighted_mass)
        
        return result, credibility

# Fonction pour afficher proprement les résultats
def print_mass(mass, label=""):
    """Affiche une fonction de masse"""
    if label:
        print(f"\n{label}:")
    for hyp, value in sorted(mass.items()):
        print(f"  m({hyp}) = {value:.6f}")

# Fonction pour normaliser les masses
def normalize_mass(mass):
    """Normalise une masse pour que la somme soit 1"""
    total = sum(mass.values())
    if total > 0:
        return {k: v/total for k, v in mass.items()}
    return mass

def verify_table2():
    """
    Vérifie les résultats de la Table 2 du papier
    """
    print("="*80)
    print("VÉRIFICATION DE LA TABLE 2")
    print("="*80)
    
    # Définition du cadre de discernement
    frame = ['A', 'B', 'C']
    ds = DSTheory(frame)
    
    # Définition des masses des 5 capteurs
    masses = [
        {'A': 0.41, 'B': 0.29, 'C': 0.3},      # S1
        {'A': 0.0, 'B': 0.9, 'C': 0.1},        # S2
        {'A': 0.58, 'B': 0.07, 'AC': 0.35},    # S3
        {'A': 0.55, 'B': 0.1, 'AC': 0.35},     # S4
        {'A': 0.6, 'B': 0.1, 'AC': 0.3}        # S5
    ]
    
    # Test des combinaisons successives
    combinations_to_test = [
        ([0, 1], "m1 ⊕ m2"),
        ([0, 1, 2], "m1 ⊕ m2 ⊕ m3"),
        ([0, 1, 2, 3], "m1 ⊕ m2 ⊕ m3 ⊕ m4"),
        ([0, 1, 2, 3, 4], "m1 ⊕ m2 ⊕ m3 ⊕ m4 ⊕ m5")
    ]
    
    for indices, label in combinations_to_test:
        print(f"\n{'='*80}")
        print(f"Combinaison: {label}")
        print(f"{'='*80}")
        
        selected_masses = [masses[i] for i in indices]
        
        # 1. Dempster
        print("\n1. DEMPSTER:")
        result = selected_masses[0].copy()
        for i in range(1, len(selected_masses)):
            result, K = ds.dempster_combination(result, selected_masses[i])
        print_mass(result, "Résultat Dempster")
        
        # 2. Yager
        print("\n2. YAGER:")
        result = selected_masses[0].copy()
        for i in range(1, len(selected_masses)):
            result = ds.yager_combination(result, selected_masses[i])
        print_mass(result, "Résultat Yager")
        
        # 3. Murphy
        print("\n3. MURPHY:")
        result = ds.murphy_combination(selected_masses)
        print_mass(result, "Résultat Murphy")
        
        # 4. Deng et al.
        print("\n4. DENG ET AL.:")
        # Implémentation simplifiée de Deng
        credibility = ds.calculate_credibility(selected_masses)
        weighted = ds.weighted_average_mass(selected_masses, credibility)
        result = ds.murphy_combination([weighted] * len(selected_masses))
        print_mass(result, "Résultat Deng et al.")
        
        # 5. Méthode proposée
        print("\n5. MÉTHODE PROPOSÉE:")
        result, cred = ds.proposed_method(selected_masses)
        print(f"Crédibilités: {cred}")
        print_mass(result, "Résultat Méthode Proposée")

def verify_table3():
    """
    Vérifie les résultats de la Table 3 du papier
    """
    print("\n" + "="*80)
    print("VÉRIFICATION DE LA TABLE 3")
    print("="*80)
    
    frame = ['A', 'B', 'C']
    ds = DSTheory(frame)
    
    # Groupe 1: Conflit élevé
    print("\n" + "-"*80)
    print("GROUPE 1: Conflit élevé")
    print("-"*80)
    masses_g1 = [
        {'A': 0.99, 'B': 0.01},
        {'B': 0.01, 'C': 0.99}
    ]
    result, cred = ds.proposed_method(masses_g1)
    print_mass(result, "Résultat Groupe 1")
    print(f"Attendu: m(A)=0.4999, m(B)=0.0002, m(C)=0.4999")
    
    # Groupe 2: Conflit faible
    print("\n" + "-"*80)
    print("GROUPE 2: Conflit faible")
    print("-"*80)
    masses_g2 = [
        {'AB': 0.5, 'C': 0.5},
        {'A': 0.5, 'BC': 0.5}
    ]
    result, cred = ds.proposed_method(masses_g2)
    print_mass(result, "Résultat Groupe 2")
    print(f"Attendu: m(A)=0.3, m(B)=0.2, m(AB)=0.1, m(C)=0.3, m(BC)=0.1")
    
    # Groupe 3: Trois évidences avec conflit élevé
    print("\n" + "-"*80)
    print("GROUPE 3: Trois évidences, conflit élevé")
    print("-"*80)
    masses_g3 = [
        {'A': 0.99, 'B': 0.01},
        {'B': 0.01, 'C': 0.99},
        {'B': 0.99, 'C': 0.01}
    ]
    result, cred = ds.proposed_method(masses_g3)
    print_mass(result, "Résultat Groupe 3")
    print(f"Attendu: m(A)≈0.313, m(B)≈0.352, m(C)≈0.332")

def analyze_conflict():
    """
    Analyse le degré de conflit entre les évidences
    """
    print("\n" + "="*80)
    print("ANALYSE DU DEGRÉ DE CONFLIT")
    print("="*80)
    
    frame = ['A', 'B', 'C']
    ds = DSTheory(frame)
    
    masses = [
        {'A': 0.41, 'B': 0.29, 'C': 0.3},      # S1
        {'A': 0.0, 'B': 0.9, 'C': 0.1},        # S2
        {'A': 0.58, 'B': 0.07, 'AC': 0.35},    # S3
        {'A': 0.55, 'B': 0.1, 'AC': 0.35},     # S4
        {'A': 0.6, 'B': 0.1, 'AC': 0.3}        # S5
    ]
    
    labels = ['S1', 'S2', 'S3', 'S4', 'S5']
    
    print("\nMatrice de similarité (cosinus):")
    S = ds.build_correlation_matrix(masses)
    
    # Affichage sous forme de table
    print("\n    ", end="")
    for label in labels:
        print(f"{label:>8}", end="")
    print()
    
    for i, label_i in enumerate(labels):
        print(f"{label_i:>4}", end="")
        for j in range(len(labels)):
            print(f"{S[i][j]:>8.4f}", end="")
        print()
    
    print("\nDegré de conflit entre les paires:")
    for i in range(len(masses)):
        for j in range(i+1, len(masses)):
            conflict = ds.conflict_degree(masses[i], masses[j])
            similarity = ds.cosine_similarity(masses[i], masses[j])
            print(f"{labels[i]} vs {labels[j]}: "
                  f"Similarité = {similarity:.4f}, "
                  f"Conflit = {conflict:.4f}")
    
    # Calcul des crédibilités
    print("\nVecteur de crédibilité:")
    cred = ds.calculate_credibility(masses)
    for i, label in enumerate(labels):
        print(f"  Crd({label}) = {cred[i]:.6f}")




# Exécuter tous les tests
if __name__ == "__main__":
    print("DÉBUT DE LA VÉRIFICATION\n")
    
    # Vérifier Table 2
    verify_table2()
    
    # Vérifier Table 3
    verify_table3()
    
    # Analyser les conflits
    analyze_conflict()
    
    
    print("\n" + "="*80)
    print("FIN DE LA VÉRIFICATION")
    print("="*80)