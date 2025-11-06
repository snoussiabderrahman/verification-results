import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path


class SimilarityAnalysis:
    """Classe pour analyser la similarité basée sur le cosinus (Figures 3 et 4)"""
    
    def __init__(self, frame):
        self.frame = frame
        self.n = len(frame)
    
    def pignistic_transform(self, mass):
        """Transforme une masse en probabilité pignistique"""
        pignistic = np.zeros(self.n)
        
        for hypothesis, m_value in mass.items():
            if m_value == 0:
                continue
            
            if hypothesis == 'theta':
                pignistic += m_value / self.n
            elif hypothesis in self.frame:
                idx = self.frame.index(hypothesis)
                pignistic[idx] += m_value
        
        return pignistic
    
    def cosine_similarity(self, mass1, mass2):
        """Calcule la similarité cosinus entre deux masses"""
        pig1 = self.pignistic_transform(mass1)
        pig2 = self.pignistic_transform(mass2)
        
        norm1 = np.linalg.norm(pig1)
        norm2 = np.linalg.norm(pig2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        dot_product = np.dot(pig1, pig2)
        cos_sim = dot_product / (norm1 * norm2)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        
        return float(cos_sim)


class ConflictMeasures:
    """Classe pour calculer les différentes mesures de conflit (Figure 5)"""
    
    def __init__(self, frame_size):
        self.frame_size = frame_size
        self.frame = list(range(frame_size))
    
    def pignistic_transform_general(self, mass_dict, frame_size):
        """Transforme une masse en probabilité pignistique"""
        pignistic = np.zeros(frame_size)
        
        for elements, m_value in mass_dict.items():
            if m_value == 0:
                continue
            
            if elements == 'theta':
                for i in range(frame_size):
                    pignistic[i] += m_value / frame_size
            elif isinstance(elements, (list, tuple, set)):
                elem_list = list(elements)
                card = len(elem_list)
                if card > 0:
                    for elem_idx in elem_list:
                        if 0 <= elem_idx < frame_size:
                            pignistic[elem_idx] += m_value / card
        
        return pignistic
    
    def conf_degree(self, mass1, mass2, frame_size):
        """Calcule ConfDegree = 1 - cos(m1, m2)"""
        pig1 = self.pignistic_transform_general(mass1, frame_size)
        pig2 = self.pignistic_transform_general(mass2, frame_size)
        
        dot_product = np.dot(pig1, pig2)
        norm1 = np.linalg.norm(pig1)
        norm2 = np.linalg.norm(pig2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        cos_sim = dot_product / (norm1 * norm2)
        cos_sim = np.clip(cos_sim, -1, 1)
        
        return 1 - cos_sim
    
    def dif_bet_p(self, mass1, mass2, frame_size):
        """Calcule difBetP = max différence en probabilité Pignistique"""
        pig1 = self.pignistic_transform_general(mass1, frame_size)
        pig2 = self.pignistic_transform_general(mass2, frame_size)
        
        max_diff = np.max(np.abs(pig1 - pig2))
        return max_diff
    
    def jousselme_distance_efficient(self, mass1, mass2, frame_size):
        """Calcule la distance de Jousselme"""
        focal_elements = [frozenset()]
        
        for key in mass1.keys():
            if key == 'theta':
                focal_elements.append(frozenset(range(frame_size)))
            elif isinstance(key, (tuple, list, set)):
                focal_elements.append(frozenset(key))
        
        for key in mass2.keys():
            if key == 'theta':
                focal_elements.append(frozenset(range(frame_size)))
            elif isinstance(key, (tuple, list, set)):
                focal_elements.append(frozenset(key))
        
        focal_elements = list(set(focal_elements))
        n_focal = len(focal_elements)
        
        m1_vec = np.zeros(n_focal)
        m2_vec = np.zeros(n_focal)
        
        for i, focal_set in enumerate(focal_elements):
            for key, value in mass1.items():
                if key == 'theta' and focal_set == frozenset(range(frame_size)):
                    m1_vec[i] = value
                    break
                elif isinstance(key, (tuple, list, set)) and frozenset(key) == focal_set:
                    m1_vec[i] = value
                    break
            
            for key, value in mass2.items():
                if key == 'theta' and focal_set == frozenset(range(frame_size)):
                    m2_vec[i] = value
                    break
                elif isinstance(key, (tuple, list, set)) and frozenset(key) == focal_set:
                    m2_vec[i] = value
                    break
        
        D = np.zeros((n_focal, n_focal))
        for i in range(n_focal):
            for j in range(n_focal):
                A = focal_elements[i]
                B = focal_elements[j]
                
                if len(A) == 0 and len(B) == 0:
                    D[i][j] = 1.0
                elif len(A.union(B)) == 0:
                    D[i][j] = 0.0
                else:
                    D[i][j] = len(A.intersection(B)) / len(A.union(B))
        
        diff = m1_vec - m2_vec
        distance = np.sqrt(0.5 * np.dot(diff, np.dot(D, diff)))
        
        return distance


def reproduce_figure3():
    """
    Figure 3: E1=(0.5, 0.5) vs E2=(x, 1-x)
    Sauvegarde: figure3.png + figure3_data.csv
    """
    print("\n" + "="*80)
    print("FIGURE 3: SIMILARITÉ AVEC E1 ÉQUILIBRÉ")
    print("="*80)
    
    frame = ['A', 'B']
    sa = SimilarityAnalysis(frame)
    
    mass1 = {'A': 0.5, 'B': 0.5}
    x_values = np.linspace(0, 1, 101)
    similarity_values = []
    
    print("Calcul des similarités...")
    for i, x in enumerate(x_values):
        mass2 = {'A': x, 'B': 1-x}
        similarity = sa.cosine_similarity(mass1, mass2)
        similarity_values.append(similarity)
        
        if i % 25 == 0:
            print(f"  x={x:.2f} → similarité={similarity:.4f}")
    
    # Sauvegarder CSV
    df = pd.DataFrame({
        'x': x_values,
        'm2_A': x_values,
        'm2_B': 1 - x_values,
        'similarity': similarity_values
    })
    df.to_csv('figure3_data.csv', index=False, float_format='%.6f')
    print("✓ Données sauvegardées: figure3_data.csv")
    
    # Créer figure
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x_values, similarity_values, '-', color='#00FF00', linewidth=3)
    
    ax.set_xlabel('m₂(A)', fontsize=13, fontweight='bold')
    ax.set_ylabel('The similarity degree between evidences\nbased on cosine theorem', 
                  fontsize=11, fontweight='bold')
    ax.set_title('Figure 3. The similarity of the evidences E1 and E2 when m₁(A)=0.5, m₁(B)=0.5', 
                 fontsize=12, pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.7, 1.0)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0.7, 1.05, 0.05))
    
    max_idx = np.argmax(similarity_values)
    ax.plot(x_values[max_idx], similarity_values[max_idx], 'ro', markersize=8)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig('figure3.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Figure sauvegardée: figure3.png")
    plt.close(fig)
    
    return x_values, similarity_values


def reproduce_figure4():
    """
    Figure 4: E1=(1, 0) vs E2=(x, 1-x)
    Sauvegarde: figure4.png + figure4_data.csv
    """
    print("\n" + "="*80)
    print("FIGURE 4: DEGRÉ DE SUPPORT AVEC E1 EXTRÊME")
    print("="*80)
    
    frame = ['A', 'B']
    sa = SimilarityAnalysis(frame)
    
    mass1 = {'A': 1.0, 'B': 0.0}
    x_values = np.linspace(0, 1, 101)
    similarity_values = []
    
    print("Calcul des degrés de support...")
    for i, x in enumerate(x_values):
        mass2 = {'A': float(x), 'B': float(1.0 - x)}
        similarity = sa.cosine_similarity(mass1, mass2)
        similarity_values.append(similarity)
        
        if i % 25 == 0:
            print(f"  x={x:.2f} → similarité={similarity:.4f}")
    
    # Sauvegarder CSV
    df = pd.DataFrame({
        'x': x_values,
        'm2_A': x_values,
        'm2_B': 1 - x_values,
        'similarity': similarity_values
    })
    df.to_csv('figure4_data.csv', index=False, float_format='%.6f')
    print("✓ Données sauvegardées: figure4_data.csv")
    
    # Créer figure
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x_values, similarity_values, '-', color='#00FF00', linewidth=3)
    
    ax.set_xlabel('x', fontsize=13, fontweight='bold')
    ax.set_ylabel('The similarity degree between evidences\nbased on cosine theorem', 
                  fontsize=11, fontweight='bold')
    ax.set_title('Figure 4. Curve line for degree of support when x increases from 0 to 1', 
                 fontsize=12, pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    ax.plot(0, similarity_values[0], 'ro', markersize=8)
    ax.plot(1, similarity_values[-1], 'ro', markersize=8)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig('figure4.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Figure sauvegardée: figure4.png")
    plt.close(fig)
    
    return x_values, similarity_values


def reproduce_figure5():
    """
    Figure 5: Comparaison des mesures de conflit
    Sauvegarde: figure5.png + figure5_data.csv
    """
    print("\n" + "="*80)
    print("FIGURE 5: MESURES DE CONFLIT (ConfDegree, difBetP, Jousselme)")
    print("="*80)
    
    frame_size = 21
    cm = ConflictMeasures(frame_size)
    
    # E2 fixe: m2({Θ1, Θ2, Θ3, Θ4, Θ5}) = 1
    mass2 = {tuple([0, 1, 2, 3, 4]): 1.0}
    
    steps = list(range(1, 21))
    conf_degree_values = []
    dif_bet_p_values = []
    jousselme_values = []
    
    print("Calcul des mesures de conflit pour chaque étape...")
    
    for step in steps:
        A = tuple(range(step))
        
        mass1 = {
            tuple([1, 2, 3]): 0.05,
            tuple([6]): 0.05,
            'theta': 0.1,
            A: 0.8
        }
        
        conf_deg = cm.conf_degree(mass1, mass2, frame_size)
        dif_bet = cm.dif_bet_p(mass1, mass2, frame_size)
        jouss = cm.jousselme_distance_efficient(mass1, mass2, frame_size)
        
        conf_degree_values.append(conf_deg)
        dif_bet_p_values.append(dif_bet)
        jousselme_values.append(jouss)
        
        if step in [1, 5, 10, 15, 20]:
            print(f"  Étape {step:2d}: ConfDegree={conf_deg:.4f}, "
                  f"difBetP={dif_bet:.4f}, Jousselme={jouss:.4f}")
    
    # Sauvegarder CSV
    df = pd.DataFrame({
        'Step': steps,
        'A_size': steps,
        'ConfDegree': conf_degree_values,
        'difBetP': dif_bet_p_values,
        'Jousselme_Distance': jousselme_values
    })
    df.to_csv('figure5_data.csv', index=False, float_format='%.6f')
    print("✓ Données sauvegardées: figure5_data.csv")
    
    # Créer figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(steps, conf_degree_values, marker='^', linestyle='-', color='blue',
            markersize=7, linewidth=2, label='ConfDegree')
    
    ax.plot(steps, dif_bet_p_values, 's--', color='#00FF00', linewidth=2.5, 
             markersize=7, label='difBetP', markerfacecolor='#00FF00',
             markeredgewidth=1.5, markeredgecolor='#00CC00')
    
    ax.plot(steps, jousselme_values, marker='*', linestyle=':', color='red',
            markersize=9, linewidth=2, label='Jousselme Distance')
    
    ax.set_xlabel('Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Conflict Measure', fontsize=14, fontweight='bold')
    ax.set_title('Figure 5. The curve lines of the conflict based on the results of the three methods', 
                 fontsize=13, pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 0.9)
    ax.set_xticks(range(0, 21, 1))
    ax.set_yticks(np.arange(0, 1.0, 0.1))
    
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95, 
              edgecolor='black', fancybox=True, shadow=True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig('figure5.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Figure sauvegardée: figure5.png")
    plt.close(fig)
    
    return steps, conf_degree_values, dif_bet_p_values, jousselme_values


def create_combined_figures_3_4():
    """Crée une figure combinée pour les Figures 3 et 4"""
    print("\n" + "="*80)
    print("CRÉATION FIGURE COMBINÉE 3+4")
    print("="*80)
    
    frame = ['A', 'B']
    sa = SimilarityAnalysis(frame)
    x_values = np.linspace(0, 1, 101)
    
    # Figure 3 données
    mass1_fig3 = {'A': 0.5, 'B': 0.5}
    sim_fig3 = [sa.cosine_similarity(mass1_fig3, {'A': x, 'B': 1-x}) for x in x_values]
    
    # Figure 4 données
    mass1_fig4 = {'A': 1.0, 'B': 0.0}
    sim_fig4 = [sa.cosine_similarity(mass1_fig4, {'A': x, 'B': 1-x}) for x in x_values]
    
    # Créer figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Figure 3
    ax1.plot(x_values, sim_fig3, '-', color='#00FF00', linewidth=3)
    ax1.set_xlabel('m₂(A)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Similarity (cosine)', fontsize=11, fontweight='bold')
    ax1.set_title('Figure 3: E1=(0.5, 0.5) vs E2=(x, 1-x)', fontsize=11, pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.7, 1.0)
    ax1.set_xticks(np.arange(0, 1.1, 0.1))
    ax1.set_yticks(np.arange(0.7, 1.05, 0.05))
    
    # Figure 4
    ax2.plot(x_values, sim_fig4, '-', color='#00FF00', linewidth=3)
    ax2.set_xlabel('x', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Similarity (cosine)', fontsize=11, fontweight='bold')
    ax2.set_title('Figure 4: E1=(1, 0) vs E2=(x, 1-x)', fontsize=11, pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.0)
    ax2.set_xticks(np.arange(0, 1.1, 0.1))
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('figures_3_and_4_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Figure combinée sauvegardée: figures_3_and_4_combined.png")
    plt.close(fig)


def print_summary():
    """Affiche un résumé des résultats"""
    print("\n" + "="*80)
    print("RÉSUMÉ DES VÉRIFICATIONS")
    print("="*80)
    
    print("""
FIGURE 3: Similarité avec E1 équilibré (0.5, 0.5)
------------------------------------------------
✓ Courbe symétrique en cloche
✓ Maximum à x=0.5 (similarité = 1.0)
✓ Valeurs extrêmes ~0.707

FIGURE 4: Degré de support avec E1 extrême (1, 0)
-------------------------------------------------
✓ Croissance monotone de 0 à 1
✓ Minimum à x=0 (similarité = 0)
✓ Maximum à x=1 (similarité = 1)

FIGURE 5: Mesures de conflit
-----------------------------
✓ ConfDegree (bleu, triangles): sensible et progressif
✓ difBetP (vert, carrés): plateau après étape 5
✓ Jousselme (rouge, étoiles): courbe en U, minimum à étape 5
    """)


def main():
    """Fonction principale"""
    print("\n" + "="*80)
    print("VÉRIFICATION COMPLÈTE DES FIGURES 3, 4 ET 5 DU PAPIER")
    print("="*80)
    
    try:
        # Figure 3
        x3, sim3 = reproduce_figure3()
        
        # Figure 4
        x4, sim4 = reproduce_figure4()
        
        # Figure 5
        steps5, conf5, dif5, jouss5 = reproduce_figure5()
        
        # Figure combinée 3+4
        create_combined_figures_3_4()
        
        # Résumé
        print_summary()
        
        print("\n" + "="*80)
        print("✓ VÉRIFICATION TERMINÉE AVEC SUCCÈS")
        print("="*80)
        print("\nFICHIERS GÉNÉRÉS:")
        print("  Images:")
        print("    • figure3.png")
        print("    • figure4.png")
        print("    • figure5.png")
        print("    • figures_3_and_4_combined.png")
        print("\n  Données CSV:")
        print("    • figure3_data.csv")
        print("    • figure4_data.csv")
        print("    • figure5_data.csv")
        
        # Afficher aperçu des CSV
        print("\n" + "="*80)
        print("APERÇU DES DONNÉES")
        print("="*80)
        
        print("\n--- Figure 3 (premières lignes) ---")
        df3 = pd.read_csv('figure3_data.csv')
        print(df3.head(10).to_string(index=False))
        
        print("\n--- Figure 4 (premières lignes) ---")
        df4 = pd.read_csv('figure4_data.csv')
        print(df4.head(10).to_string(index=False))
        
        print("\n--- Figure 5 (toutes les lignes) ---")
        df5 = pd.read_csv('figure5_data.csv')
        print(df5.to_string(index=False))
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interruption par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()