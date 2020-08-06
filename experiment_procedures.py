"""
    This module contains the auxiliary procedure for running the experiments
"""


def resettv(samp=1):
    """
    Setting the parameters for running the windowed experiment
    :param samp: The samp value
    :return: the parameters
    """
    parameters = {"fonollosa": {"ini_value": int(5000/5),
                                "start_value": int(6400/5),
                                "step": int(1400/5),
                                "end_value": int(19000/5)+1,
                                },
                  "windtunnel": {"ini_value": int(20*samp),
                                 "start_value": int(44*samp),
                                 "step": int(23*samp),
                                 "end_value": int(260*samp)},
                  "turbulent_gas_mixtures": {"ini_value": int(600/samp),
                                             "start_value": int(837/samp),
                                             "step": int(236/samp),
                                             "end_value": int(2970/samp)},
                  "QWines-CsystemTR": {"ini_value": int(160/samp),
                                       "start_value": int(474/samp),
                                       "step": int(314/samp),
                                       "end_value": int(3300/samp) + 1},
                  "QWinesEa-CsystemTR": {"ini_value": int(160/samp),
                                         "start_value": int(474/samp),
                                         "step": int(314/samp),
                                         "end_value": int(3300/samp) + 1}
                  # Uncomment the following lines if you have been authorized to
                  # ,"coffee_dataset": {"ini_value": int(29/samp),
                  #                   "start_value": int(56/samp),
                  #                   "step": int(26/samp),
                  #                   "end_value": int(299/samp)}
                  }
    return parameters
