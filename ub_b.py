import glob
from deeph3.util.pdb import protein_dist_angle_matrix
from deeph3.preprocess.create_antibody_db import download_and_truncate_pdbs
import matplotlib.pyplot as plt
import seaborn as sns

pairs = open("pdb_list2.txt").readlines()

pdb_directory = "data/antibody_database_redundant/"

fig_directory = "data/heatmaps/"


mat1 = []
mat2 = []

heatmap_style = {
    'center': 0,
    'square': True,
    'xticklabels': 50,
    'yticklabels': 50,
    'cmap': 'coolwarm',
    'cbar_kws': {
            "shrink": 0.75
    }
}


for i, p in enumerate(pairs):

    print("UB/B Pair:", p)

    u, b = p.strip('\n').split('\t')

    u_pdb = glob.glob(pdb_directory + u + "*.pdb")
    b_pdb = glob.glob(pdb_directory + b + "*.pdb")


    if u_pdb and b_pdb:

        unbound = u_pdb[0]
        bound = b_pdb[0]

        u_fasta =  glob.glob(pdb_directory + u + "*.fasta")
        b_fasta =  glob.glob(pdb_directory + b + "*.fasta")

        u_seq = u_fasta[0]
        b_seq = b_fasta[0]


        unbound_dist = protein_dist_angle_matrix(unbound, u_seq)[0]
        bound_dist = protein_dist_angle_matrix(bound, b_seq)[0]

        mat1.append([unbound_dist, bound_dist])

        #print(unbound.split("/")[-1])
        #htmp(unbound_dist)
        #print(bound.split("/")[-1])
        #htmp(bound_dist)
        

        if bound_dist.shape == unbound_dist.shape:
            print('*')

            mask = ((unbound_dist == -999) | (bound_dist == -999)).numpy()
            bind_dif = bound_dist- unbound_dist
            mat2.append(bind_dif)


            fig = plt.figure()
            sns.heatmap(bind_dif, mask = mask, vmin=-10, vmax=10, **heatmap_style)
            fig.patch.set_facecolor('white')
            plt.title('U: {} B: {}'.format(u,b))
            fig.savefig(fig_directory + 'tst_{}_{}_{}.png'.format(i+1,u,b), facecolor=fig.get_facecolor(),     edgecolor='none')

            print('-')

        else:
            print("Pair sizes do not match")

        print('--')

    else:
        print("Pair not found in directory") 

    print('---')

print("Pairs listed: {}".format(len(pairs)))
print("Pairs found: {}".format(len(mat1)))
print("Pairs compared: {}".format(len(mat2)))