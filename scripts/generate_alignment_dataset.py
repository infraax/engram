"""Generate alignment dataset: SVD state vectors for same docs on two models."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from llama_cpp import Llama

from kvcos.core.blob_parser import parse_state_blob
from kvcos.core.cache_spec import make_spec_from_metadata
from kvcos.core.types import StateExtractionMode
from kvcos.core.state_extractor import MARStateExtractor

# 50 diverse documents: 5 per domain × 10 domains
DOCUMENTS = [
    # ML/AI (0-4)
    "Gradient descent optimizes neural network parameters by computing partial derivatives of the loss function with respect to each weight and updating weights in the direction that reduces loss.",
    "Convolutional neural networks apply learned filters across spatial dimensions of input images, producing feature maps that detect edges, textures, and higher-level visual patterns.",
    "Recurrent neural networks process sequences by maintaining hidden state that carries information across time steps, enabling the model to capture temporal dependencies in data.",
    "Batch normalization normalizes layer inputs during training by subtracting the mini-batch mean and dividing by the mini-batch standard deviation, accelerating convergence.",
    "Dropout regularization randomly sets neuron activations to zero during training with probability p, preventing co-adaptation and reducing overfitting in deep networks.",
    # Biology (5-9)
    "Mitochondria generate ATP through oxidative phosphorylation, where electrons pass through complexes I through IV of the electron transport chain embedded in the inner membrane.",
    "Photosynthesis in chloroplasts converts carbon dioxide and water into glucose using light energy captured by chlorophyll molecules in the thylakoid membrane.",
    "The immune system distinguishes self from non-self through major histocompatibility complex proteins that present intracellular peptide fragments to T lymphocytes.",
    "Synaptic transmission involves calcium-dependent exocytosis of neurotransmitter vesicles at the presynaptic terminal followed by receptor binding at the postsynaptic membrane.",
    "Enzyme kinetics follow Michaelis-Menten dynamics where reaction velocity approaches Vmax asymptotically as substrate concentration increases relative to the Km constant.",
    # History (10-14)
    "The French Revolution of 1789 abolished feudal privileges and established principles of popular sovereignty that fundamentally altered European political structures.",
    "The Silk Road connected Chinese Han dynasty merchants with Roman traders across Central Asia, facilitating exchange of silk, spices, and metallurgical techniques.",
    "The Industrial Revolution began in eighteenth-century Britain with mechanized textile production, steam power, and factory organization transforming agrarian economies.",
    "Ancient Egyptian civilization developed hieroglyphic writing, monumental architecture, and sophisticated irrigation systems along the Nile River floodplain.",
    "The Renaissance in fifteenth-century Florence produced breakthroughs in perspective painting, humanist philosophy, and anatomical studies by artists like Leonardo.",
    # Cooking (15-19)
    "Maillard reactions between amino acids and reducing sugars at temperatures above 140 degrees Celsius produce the brown color and complex flavors of seared meat.",
    "Emulsification in mayonnaise relies on lecithin from egg yolks to stabilize the dispersion of oil droplets in the aqueous vinegar and lemon juice phase.",
    "Bread leavening occurs when Saccharomyces cerevisiae ferments sugars in dough, producing carbon dioxide gas that becomes trapped in the gluten network.",
    "Caramelization of sucrose begins at 160 degrees Celsius as the disaccharide breaks down into glucose and fructose which then undergo further dehydration.",
    "Brining meat in a salt solution denatures surface proteins and increases water retention through osmotic effects, producing juicier cooked results.",
    # Mathematics (20-24)
    "The fundamental theorem of calculus establishes that differentiation and integration are inverse operations, connecting the derivative of an integral to the original function.",
    "Eigenvalues of a square matrix A satisfy the characteristic equation det(A - lambda I) = 0, with corresponding eigenvectors spanning invariant subspaces.",
    "The central limit theorem states that the sampling distribution of the mean approaches a normal distribution as sample size increases regardless of population shape.",
    "Group theory studies algebraic structures with a binary operation satisfying closure, associativity, identity, and invertibility axioms.",
    "Fourier transforms decompose signals into constituent sinusoidal frequencies, enabling spectral analysis and convolution operations in the frequency domain.",
    # Literature (25-29)
    "Shakespeare's tragedies explore fatal character flaws: Hamlet's indecision, Macbeth's ambition, Othello's jealousy, and King Lear's prideful blindness.",
    "Stream of consciousness narration in Joyce's Ulysses follows Leopold Bloom's interior monologue through Dublin in a single day paralleling Homer's Odyssey.",
    "Magical realism in Garcia Marquez's fiction blends supernatural events with mundane Latin American reality, challenging Western rationalist literary conventions.",
    "The bildungsroman genre traces protagonist maturation from youth to adulthood, exemplified by Dickens's Great Expectations and Bronte's Jane Eyre.",
    "Haiku poetry constrains expression to seventeen syllables across three lines, using seasonal reference words to evoke natural imagery and transient emotion.",
    # Economics (30-34)
    "Supply and demand curves intersect at equilibrium price where quantity supplied equals quantity demanded, with shifts caused by external factors like income changes.",
    "Monetary policy adjusts interest rates and money supply to influence inflation, employment, and economic growth through central bank open market operations.",
    "Game theory models strategic interactions where each player's optimal decision depends on expectations about other players' choices and resulting payoff matrices.",
    "Comparative advantage explains why countries benefit from trade even when one nation produces all goods more efficiently than its trading partner.",
    "Behavioral economics incorporates psychological biases like loss aversion and anchoring into economic models, departing from purely rational agent assumptions.",
    # Physics (35-39)
    "Quantum entanglement creates correlations between particles such that measuring one instantaneously determines the state of the other regardless of separation distance.",
    "General relativity describes gravity as spacetime curvature caused by mass-energy, predicting phenomena like gravitational time dilation and black hole event horizons.",
    "Thermodynamic entropy measures disorder in a system, with the second law stating that total entropy of an isolated system can only increase over time.",
    "Superconductivity occurs below critical temperature when electron pairs form Cooper pairs that flow without resistance through the crystal lattice.",
    "The Heisenberg uncertainty principle establishes a fundamental limit on simultaneously knowing both position and momentum of a quantum particle.",
    # Geography (40-44)
    "Tectonic plate boundaries produce earthquakes at transform faults, volcanic activity at subduction zones, and new oceanic crust at mid-ocean spreading ridges.",
    "The Amazon River basin contains the largest tropical rainforest ecosystem, supporting approximately ten percent of all known species on Earth.",
    "Glacial erosion carved U-shaped valleys, cirques, and fjords during Pleistocene ice ages when ice sheets covered much of northern Europe and North America.",
    "Mediterranean climate zones occur on western continental coasts between latitudes 30 and 45 degrees, characterized by dry summers and mild wet winters.",
    "The Sahara Desert receives less than 25 millimeters of annual rainfall, with extreme diurnal temperature variation exceeding 30 degrees Celsius.",
    # Programming (45-49)
    "Hash tables provide average O(1) lookup time by mapping keys through a hash function to array indices, with collision resolution via chaining or open addressing.",
    "Garbage collection in managed runtimes automatically reclaims memory by tracing reachable objects from root references and freeing unreachable allocations.",
    "TCP ensures reliable data delivery through sequence numbers, acknowledgments, retransmission timers, and flow control using sliding window protocol.",
    "Database normalization eliminates redundancy by decomposing relations into smaller tables satisfying normal forms while preserving functional dependencies.",
    "Version control with git tracks content changes using a directed acyclic graph of commit objects, each containing a tree hash, parent references, and metadata.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate cross-model alignment dataset")
    parser.add_argument("--model-a", required=True, help="Path to model A GGUF")
    parser.add_argument("--model-b", required=True, help="Path to model B GGUF")
    parser.add_argument("--n-docs", type=int, default=50)
    parser.add_argument("--layer-range-a", type=int, nargs=2, default=[8, 24])
    parser.add_argument("--layer-range-b", type=int, nargs=2, default=[8, 24])
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()

    docs = DOCUMENTS[: args.n_docs]

    def extract_all(model_path: str, layer_range: tuple[int, int]) -> torch.Tensor:
        llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
        meta = llm.metadata
        n_layers = int(meta.get("llama.block_count", "32"))
        n_heads = int(meta.get("llama.attention.head_count", "32"))
        n_kv_heads = int(meta.get("llama.attention.head_count_kv", "8"))
        head_dim = int(meta.get("llama.embedding_length", "4096")) // n_heads
        model_name = meta.get("general.name", Path(model_path).stem)

        spec = make_spec_from_metadata(
            model_id=model_name, n_layers=n_layers, n_heads=n_heads,
            n_kv_heads=n_kv_heads, head_dim=head_dim,
        )
        ext = MARStateExtractor(
            mode=StateExtractionMode.SVD_PROJECT,
            rank=128, layer_range=layer_range, gate_start=6,
        )

        print(f"Extracting from {model_name} ({n_layers}L/{n_kv_heads}KV/{head_dim}D)...")
        vecs = []
        for i, doc in enumerate(docs):
            llm.reset()
            llm(doc.strip(), max_tokens=1, temperature=0.0)
            s = llm.save_state()
            p = parse_state_blob(bytes(s.llama_state), n_kv_heads=n_kv_heads, head_dim=head_dim)
            r = ext.extract(p.keys, spec)
            vecs.append(r.state_vec)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(docs)}")

        del llm
        gc.collect()
        return torch.stack(vecs)

    vecs_a = extract_all(args.model_a, tuple(args.layer_range_a))
    vecs_b = extract_all(args.model_b, tuple(args.layer_range_b))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vecs_a": vecs_a, "vecs_b": vecs_b, "n_docs": len(docs)}, str(output_path))
    print(f"\nSaved: {output_path} ({vecs_a.shape[0]} docs, dim_a={vecs_a.shape[1]}, dim_b={vecs_b.shape[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
