
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import math

def drawData(dataclass, BaseModel):
    prefix = 'PlotResults'
    font = {'family': 'serif', 'weight': 'normal', 'size': 12}
    matplotlib.rc('font', **font)
    LineWidth = 2
    fontsize = 10
    markersize = 3

    if dataclass == 'EPFL':
        if BaseModel == "Plenoptic":
            bpp = [0.021090441, 0.008873706, 0.006483945, 0.00493264, 0.003814983, 0.00302237, 0.002426938,
                   0.001951035, ]
            psnr = [35.91644923, 34.09243455, 33.60664075, 33.05759529, 32.46824248, 31.78869538, 31.07183964,
                    30.20359396]
            GCC, = plt.plot(bpp, psnr, "-o", color=(0.462745, 0.3137, 0.0196), linewidth=LineWidth,
                            label='GCC')

            bpp = [0.026553, 0.009875, 0.004532, 0.002298, 0.001297, ]
            psnr = [35.9815, 34.2063, 32.2788, 30.2252, 27.9053]
            SPRVVC, = plt.plot(bpp, psnr, "-o", color=(0.98039, 0.42745, 0.1197), linewidth=LineWidth,
                               label='SPR-VVC')

            bpp = [0.024518, 0.00906, 0.004145, 0.00211, 0.001204, ]
            psnr = [35.6623, 33.6512, 31.528, 29.2986, 26.7833]
            SOPVVC, = plt.plot(bpp, psnr, "-o", color=(0.0549, 0.17254, 0.5098), linewidth=LineWidth,
                               label='SOP-VVC')

            bpp = [0.039888, 0.015042, 0.006416, 0.002964, 0.001655, ]
            psnr = [35.6094, 33.7287, 31.7175, 29.5673, 27.3437]
            SPRHEVC, = plt.plot(bpp, psnr, "-o", color=(0.713725, 0.7098, 0.12156), linewidth=LineWidth,
                                label='SPR-HEVC')

            bpp = [0.032984, 0.012389, 0.005273, 0.002514, 0.001481, ]
            psnr = [35.4484, 33.3961, 31.17, 28.8304, 26.2517]
            SOPHEVC, = plt.plot(bpp, psnr, "-o", color=(0.541, 0.169, 0.886), linewidth=LineWidth,
                                label='SOP-HEVC')

            bpp = [0.018696, 0.013235, 0.0093, 0.006484, 0.004553]
            psnr = [32.1074, 31.22614, 30.3445, 29.49759, 28.55652]
            GPR, = plt.plot(bpp, psnr, "-o", color=(0.8549, 0.1333, 0.0941176), linewidth=LineWidth,
                            label='GPR')

            bpp = [0.018592, 0.010042, 0.008125, 0.0015333, 0.001283, ]
            psnr = [38.30107, 36.6172, 35.83108, 31.45201, 29.97209] ## Tensorflow 2.4  on V100
            # bpp = [0.004039657988540304,
            #       0.004882850913422905,
            #       0.0065410815534089695,
            #       0.009014525512717492,
            #       0.011603517048125858,
            #       0.02045320873394509, ]
            # psnr = [29.424056487863766,
            #       30.686476120651378,
            #       32.46693732873961,
            #       33.9203876792781,
            #       34.842683485673014,
            #       36.3234157660037,]## pytorch 2.3 on RTX3090
            Proposed, = plt.plot(bpp, psnr, "-o", color=(0.02745, 0.5019, 0.81176), linewidth=LineWidth,
                                 label='Proposed')
            plt.legend(handles=[GCC, SPRVVC, SOPVVC, SPRHEVC, SOPHEVC, GPR, Proposed], loc=4, fontsize=6)

        if BaseModel == "Learned":
            bpp = [0.019533, 0.012958, 0.007083, 0.006658]
            psnr = [33.70197, 32.44355, 28.41436, 26.9587]
            Cheng, = plt.plot(bpp, psnr, "-o", color=(0.956862, 0.47843, 0.458823), linewidth=LineWidth,
                              label='Cheng\'s')

            bpp = [0.020983333, 0.017025, 0.011542, 0.005833, ]
            psnr = [29.80256758, 29.52545, 28.61123, 22.20934]
            Minnen, = plt.plot(bpp, psnr, "-o", color=(0.007843, 0.2941176, 0.317647), linewidth=LineWidth,
                               label='Minnen\'s')
            bpp = [0.018592, 0.010042, 0.008125, 0.0015333, 0.001283, ]
            psnr = [38.30107, 36.6172, 35.83108, 31.45201, 29.97209] ## Tensorflow 2.4  on V100
            # bpp = [0.004039657988540304,
            #        0.004882850913422905,
            #        0.0065410815534089695,
            #        0.009014525512717492,
            #        0.011603517048125858,
            #        0.02045320873394509, ]
            # psnr = [29.424056487863766,
            #         30.686476120651378,
            #         32.46693732873961,
            #         33.9203876792781,
            #         34.842683485673014,
            #         36.3234157660037, ]  ## pytorch 2.3 on RTX3090
            Proposed, = plt.plot(bpp, psnr, "-o", color=(0.02745, 0.5019, 0.81176), linewidth=LineWidth,
                                 label='Proposed')
            plt.legend(handles=[Cheng, Minnen, Proposed], loc=4, fontsize=6)


    else:
        print('no such class : ' + dataclass)
        exit(0)
    savepathpsnr = prefix + '/' + dataclass + '_psnr'  # + '.eps'
    print(prefix)
    if not os.path.exists(prefix):
        os.makedirs(prefix)



    plt.rcParams.update({'font.size': 12})

    plt.grid()
    plt.xlabel('bit per pixel (bpp)')
    plt.ylabel('PSNR (dB)')
    plt.xlim(0, 0.022)
    # _, right = plt.xlim()  # return the current xlim
    # plt.xlim((0, right))  # set the xlim to left, right

    bottom, top = plt.ylim()  # return the current ylim
    plt.ylim((max(24, bottom//2*2), top))  # set the xlim to left, right
    # plt.savefig(savepathpsnr + '.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(savepathpsnr + '.png', dpi=300)
    plt.clf()

    # ----------------------------------------MSSSIM-------------------------------------------------
    if dataclass == 'EPFL':
        if BaseModel == "Plenoptic":
            bpp = [0.021090441, 0.008873706, 0.006483945, 0.00493264, 0.003814983, 0.00302237, 0.002426938,
                   0.001951035, ]
            msssim = [0.948861871, 0.936551414, 0.928662462, 0.918925836, 0.907383407, 0.893122486, 0.876219445,
                      0.857384167]
            GCC, = plt.plot(bpp, -10 * np.log10(np.subtract(1, msssim)), "-o", color=(0.462745, 0.3137, 0.0196),
                            linewidth=LineWidth,
                            label='GCC')

            bpp = [0.026553, 0.009875, 0.004532, 0.002298, 0.001297, ]
            msssim = [0.946614, 0.925727, 0.894208, 0.850555, 0.787523]
            SPRVVC, = plt.plot(bpp, -10 * np.log10(np.subtract(1, msssim)), "-o", color=(0.98039, 0.42745, 0.1197),
                               linewidth=LineWidth,
                               label='SPR-VVC')

            bpp = [0.024518, 0.00906, 0.004145, 0.00211, 0.001204, ]
            msssim = [0.939763, 0.911237, 0.871304, 0.816267, 0.735391]
            SOPVVC, = plt.plot(bpp, -10 * np.log10(np.subtract(1, msssim)), "-o", color=(0.0549, 0.17254, 0.5098),
                               linewidth=LineWidth,
                               label='SOP-VVC')

            bpp = [0.039888, 0.015042, 0.006416, 0.002964, 0.001655, ]
            msssim = [0.941812, 0.917458, 0.881208, 0.832309, 0.767596]
            SPRHEVC, = plt.plot(bpp, -10 * np.log10(np.subtract(1, msssim)), "-o", color=(0.713725, 0.7098, 0.12156),
                                linewidth=LineWidth,
                                label='SPR-HEVC')

            bpp = [0.032984, 0.012389, 0.005273, 0.002514, 0.001481, ]
            msssim = [0.938011, 0.908317, 0.86384, 0.805366, 0.720302]
            SOPHEVC, = plt.plot(bpp, -10 * np.log10(np.subtract(1, msssim)), "-o", color=(0.541, 0.169, 0.886),
                                linewidth=LineWidth,
                                label='SOP-HEVC')

            bpp = [0.018696, 0.013235, 0.0093, 0.006484, 0.004553, ]
            msssim = [0.902919, 0.887071, 0.868255, 0.846783, 0.82067]
            GPR, = plt.plot(bpp, -10 * np.log10(np.subtract(1, msssim)), "-o", color=(0.8549, 0.1333, 0.0941176),
                            linewidth=LineWidth,
                            label='GPR')

            bpp = [0.018592, 0.0100421, 0.008125, 0.001533, 0.001283, ]
            msssim = [0.971556, 0.96214, 0.956156, 0.9001169, 0.844893] ## Tensorflow 2.4  on V100
            # bpp = [0.004039657988540304,
            #        0.004882850913422905,
            #        0.0065410815534089695,
            #        0.009014525512717492,
            #        0.011603517048125858,
            #        0.02045320873394509, ]
            # msssim = [0.915062472,
            #         0.931877578,
            #         0.951208179,
            #         0.964940732,
            #         0.972166126,
            #         0.981831407] ## pytorch 2.3 on RTX3090
            Proposed, = plt.plot(bpp, -10 * np.log10(np.subtract(1, msssim)), "-o", color=(0.02745, 0.5019, 0.81176),
                                 linewidth=LineWidth,
                                 label='Proposed')

            plt.legend(handles=[GCC, SPRVVC, SOPVVC, SPRHEVC, SOPHEVC, GPR, Proposed], loc=4,
                       fontsize=6)

        if BaseModel == "Learned":
            bpp = [0.019533, 0.012958333, 0.007083333, 0.006658, ]
            msssim = [0.926442, 0.904079833, 0.7800995, 0.7113055]
            Cheng, = plt.plot(bpp, -10 * np.log10(np.subtract(1, msssim)), "-o", color=(0.956862, 0.47843, 0.458823),
                              linewidth=LineWidth,
                              label='Cheng\'s')
            bpp = [0.020983333, 0.017025, 0.011542, 0.005833, ]
            msssim = [0.837900333, 0.82231, 0.804488, 0.645613]
            Minnen, = plt.plot(bpp, -10 * np.log10(np.subtract(1, msssim)), "-o", color=(0.007843, 0.2941176, 0.317647),
                               linewidth=LineWidth,
                               label='Minnen\'s')

            bpp = [0.018592, 0.0100421, 0.008125, 0.001533, 0.001283, ]
            msssim = [0.971556, 0.96214, 0.956156, 0.9001169, 0.844893] ## Tensorflow 2.4  on V100
            # bpp = [0.004039657988540304,
            #        0.004882850913422905,
            #        0.0065410815534089695,
            #        0.009014525512717492,
            #        0.011603517048125858,
            #        0.02045320873394509, ]
            # msssim = [0.915062472,
            #           0.931877578,
            #           0.951208179,
            #           0.964940732,
            #           0.972166126,
            #           0.981831407]  ## pytorch 2.3 on RTX3090
            Proposed, = plt.plot(bpp, -10 * np.log10(np.subtract(1, msssim)), "-o", color=(0.02745, 0.5019, 0.81176),
                                 linewidth=LineWidth,
                                 label='Proposed')

            plt.legend(handles=[Cheng, Minnen, Proposed], loc=4, fontsize=6)
            plt.xlim(0, 0.022)

    else:
        print('no such class : ', + dataclass)
        exit(0)

    savepathmsssim = prefix + '/' + dataclass + '_msssim'  # + '.eps'

    plt.rcParams.update({'font.size': 12})
    plt.xlim(0, 0.022)
    # left, right = plt.xlim()  # return the current xlim
    # plt.xlim((0, right))  # set the xlim to left, right
    plt.grid()
    plt.xlabel('bit per pixel (bpp)')
    plt.ylabel('MS-SSIM (dB)')

    # plt.savefig(savepathmsssim + '.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(savepathmsssim + '.png', dpi=300)
    plt.clf()

    savepath = prefix + '/' + dataclass + '.png'
    img1 = cv2.imread(savepathpsnr + '.png')
    img2 = cv2.imread(savepathmsssim + '.png')

    image = np.concatenate((img1, img2), axis=1)
    cv2.imwrite(savepath, image)



drawData('EPFL', 'Plenoptic')
# drawData('EPFL', 'Learned')



