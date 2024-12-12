from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch

# Initialiser la barre de progression
tqdm.pandas()

def BERTweet_embedding_minute(raw_folder, processed_folder, embedded_folder):
    # Charger le tokenizer et le modèle BERTweet
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    model = AutoModel.from_pretrained("vinai/bertweet-base")

    # Charger les données brutes
    file_path = f"data/processed_data/{processed_folder}/{raw_folder}.csv"
    df = pd.read_csv(file_path)

    print("Transforming tweets with BERTweet...")
    # Générer les embeddings
    df['embeddings'] = df['Tweet'].progress_apply(lambda x: get_bertweet_embeddings(x, tokenizer, model))

    # Convertir les embeddings en colonnes
    embedding_df = pd.DataFrame(df['embeddings'].tolist(), index=df.index)
    embedding_df.columns = [f"dim_{i+1}" for i in range(embedding_df.shape[1])]

    # Ajouter les embeddings au DataFrame
    df = pd.concat([df, embedding_df], axis=1)

    # Préparer le DataFrame final
    period_features = df.drop(columns=['Timestamp', 'Tweet', 'embeddings'])
    print(period_features)
    str = "'MatchID,PeriodID,EventType,dim_1,dim_2,dim_3,dim_4,dim_5,dim_6,dim_7,dim_8,dim_9,dim_10,dim_11,dim_12,dim_13,dim_14,dim_15,dim_16,dim_17,dim_18,dim_19,dim_20,dim_21,dim_22,dim_23,dim_24,dim_25,dim_26,dim_27,dim_28,dim_29,dim_30,dim_31,dim_32,dim_33,dim_34,dim_35,dim_36,dim_37,dim_38,dim_39,dim_40,dim_41,dim_42,dim_43,dim_44,dim_45,dim_46,dim_47,dim_48,dim_49,dim_50,dim_51,dim_52,dim_53,dim_54,dim_55,dim_56,dim_57,dim_58,dim_59,dim_60,dim_61,dim_62,dim_63,dim_64,dim_65,dim_66,dim_67,dim_68,dim_69,dim_70,dim_71,dim_72,dim_73,dim_74,dim_75,dim_76,dim_77,dim_78,dim_79,dim_80,dim_81,dim_82,dim_83,dim_84,dim_85,dim_86,dim_87,dim_88,dim_89,dim_90,dim_91,dim_92,dim_93,dim_94,dim_95,dim_96,dim_97,dim_98,dim_99,dim_100,dim_101,dim_102,dim_103,dim_104,dim_105,dim_106,dim_107,dim_108,dim_109,dim_110,dim_111,dim_112,dim_113,dim_114,dim_115,dim_116,dim_117,dim_118,dim_119,dim_120,dim_121,dim_122,dim_123,dim_124,dim_125,dim_126,dim_127,dim_128,dim_129,dim_130,dim_131,dim_132,dim_133,dim_134,dim_135,dim_136,dim_137,dim_138,dim_139,dim_140,dim_141,dim_142,dim_143,dim_144,dim_145,dim_146,dim_147,dim_148,dim_149,dim_150,dim_151,dim_152,dim_153,dim_154,dim_155,dim_156,dim_157,dim_158,dim_159,dim_160,dim_161,dim_162,dim_163,dim_164,dim_165,dim_166,dim_167,dim_168,dim_169,dim_170,dim_171,dim_172,dim_173,dim_174,dim_175,dim_176,dim_177,dim_178,dim_179,dim_180,dim_181,dim_182,dim_183,dim_184,dim_185,dim_186,dim_187,dim_188,dim_189,dim_190,dim_191,dim_192,dim_193,dim_194,dim_195,dim_196,dim_197,dim_198,dim_199,dim_200,dim_201,dim_202,dim_203,dim_204,dim_205,dim_206,dim_207,dim_208,dim_209,dim_210,dim_211,dim_212,dim_213,dim_214,dim_215,dim_216,dim_217,dim_218,dim_219,dim_220,dim_221,dim_222,dim_223,dim_224,dim_225,dim_226,dim_227,dim_228,dim_229,dim_230,dim_231,dim_232,dim_233,dim_234,dim_235,dim_236,dim_237,dim_238,dim_239,dim_240,dim_241,dim_242,dim_243,dim_244,dim_245,dim_246,dim_247,dim_248,dim_249,dim_250,dim_251,dim_252,dim_253,dim_254,dim_255,dim_256,dim_257,dim_258,dim_259,dim_260,dim_261,dim_262,dim_263,dim_264,dim_265,dim_266,dim_267,dim_268,dim_269,dim_270,dim_271,dim_272,dim_273,dim_274,dim_275,dim_276,dim_277,dim_278,dim_279,dim_280,dim_281,dim_282,dim_283,dim_284,dim_285,dim_286,dim_287,dim_288,dim_289,dim_290,dim_291,dim_292,dim_293,dim_294,dim_295,dim_296,dim_297,dim_298,dim_299,dim_300,dim_301,dim_302,dim_303,dim_304,dim_305,dim_306,dim_307,dim_308,dim_309,dim_310,dim_311,dim_312,dim_313,dim_314,dim_315,dim_316,dim_317,dim_318,dim_319,dim_320,dim_321,dim_322,dim_323,dim_324,dim_325,dim_326,dim_327,dim_328,dim_329,dim_330,dim_331,dim_332,dim_333,dim_334,dim_335,dim_336,dim_337,dim_338,dim_339,dim_340,dim_341,dim_342,dim_343,dim_344,dim_345,dim_346,dim_347,dim_348,dim_349,dim_350,dim_351,dim_352,dim_353,dim_354,dim_355,dim_356,dim_357,dim_358,dim_359,dim_360,dim_361,dim_362,dim_363,dim_364,dim_365,dim_366,dim_367,dim_368,dim_369,dim_370,dim_371,dim_372,dim_373,dim_374,dim_375,dim_376,dim_377,dim_378,dim_379,dim_380,dim_381,dim_382,dim_383,dim_384,dim_385,dim_386,dim_387,dim_388,dim_389,dim_390,dim_391,dim_392,dim_393,dim_394,dim_395,dim_396,dim_397,dim_398,dim_399,dim_400,dim_401,dim_402,dim_403,dim_404,dim_405,dim_406,dim_407,dim_408,dim_409,dim_410,dim_411,dim_412,dim_413,dim_414,dim_415,dim_416,dim_417,dim_418,dim_419,dim_420,dim_421,dim_422,dim_423,dim_424,dim_425,dim_426,dim_427,dim_428,dim_429,dim_430,dim_431,dim_432,dim_433,dim_434,dim_435,dim_436,dim_437,dim_438,dim_439,dim_440,dim_441,dim_442,dim_443,dim_444,dim_445,dim_446,dim_447,dim_448,dim_449,dim_450,dim_451,dim_452,dim_453,dim_454,dim_455,dim_456,dim_457,dim_458,dim_459,dim_460,dim_461,dim_462,dim_463,dim_464,dim_465,dim_466,dim_467,dim_468,dim_469,dim_470,dim_471,dim_472,dim_473,dim_474,dim_475,dim_476,dim_477,dim_478,dim_479,dim_480,dim_481,dim_482,dim_483,dim_484,dim_485,dim_486,dim_487,dim_488,dim_489,dim_490,dim_491,dim_492,dim_493,dim_494,dim_495,dim_496,dim_497,dim_498,dim_499,dim_500,dim_501,dim_502,dim_503,dim_504,dim_505,dim_506,dim_507,dim_508,dim_509,dim_510,dim_511,dim_512,dim_513,dim_514,dim_515,dim_516,dim_517,dim_518,dim_519,dim_520,dim_521,dim_522,dim_523,dim_524,dim_525,dim_526,dim_527,dim_528,dim_529,dim_530,dim_531,dim_532,dim_533,dim_534,dim_535,dim_536,dim_537,dim_538,dim_539,dim_540,dim_541,dim_542,dim_543,dim_544,dim_545,dim_546,dim_547,dim_548,dim_549,dim_550,dim_551,dim_552,dim_553,dim_554,dim_555,dim_556,dim_557,dim_558,dim_559,dim_560,dim_561,dim_562,dim_563,dim_564,dim_565,dim_566,dim_567,dim_568,dim_569,dim_570,dim_571,dim_572,dim_573,dim_574,dim_575,dim_576,dim_577,dim_578,dim_579,dim_580,dim_581,dim_582,dim_583,dim_584,dim_585,dim_586,dim_587,dim_588,dim_589,dim_590,dim_591,dim_592,dim_593,dim_594,dim_595,dim_596,dim_597,dim_598,dim_599,dim_600,dim_601,dim_602,dim_603,dim_604,dim_605,dim_606,dim_607,dim_608,dim_609,dim_610,dim_611,dim_612,dim_613,dim_614,dim_615,dim_616,dim_617,dim_618,dim_619,dim_620,dim_621,dim_622,dim_623,dim_624,dim_625,dim_626,dim_627,dim_628,dim_629,dim_630,dim_631,dim_632,dim_633,dim_634,dim_635,dim_636,dim_637,dim_638,dim_639,dim_640,dim_641,dim_642,dim_643,dim_644,dim_645,dim_646,dim_647,dim_648,dim_649,dim_650,dim_651,dim_652,dim_653,dim_654,dim_655,dim_656,dim_657,dim_658,dim_659,dim_660,dim_661,dim_662,dim_663,dim_664,dim_665,dim_666,dim_667,dim_668,dim_669,dim_670,dim_671,dim_672,dim_673,dim_674,dim_675,dim_676,dim_677,dim_678,dim_679,dim_680,dim_681,dim_682,dim_683,dim_684,dim_685,dim_686,dim_687,dim_688,dim_689,dim_690,dim_691,dim_692,dim_693,dim_694,dim_695,dim_696,dim_697,dim_698,dim_699,dim_700,dim_701,dim_702,dim_703,dim_704,dim_705,dim_706,dim_707,dim_708,dim_709,dim_710,dim_711,dim_712,dim_713,dim_714,dim_715,dim_716,dim_717,dim_718,dim_719,dim_720,dim_721,dim_722,dim_723,dim_724,dim_725,dim_726,dim_727,dim_728,dim_729,dim_730,dim_731,dim_732,dim_733,dim_734,dim_735,dim_736,dim_737,dim_738,dim_739,dim_740,dim_741,dim_742,dim_743,dim_744,dim_745,dim_746,dim_747,dim_748,dim_749,dim_750,dim_751,dim_752,dim_753,dim_754,dim_755,dim_756,dim_757,dim_758,dim_759,dim_760,dim_761,dim_762,dim_763,dim_764,dim_765,dim_766,dim_767,dim_768'"
    str = str.replace(",", "', '")
    final_df = period_features.groupby('ID')[str].mean()

    # Sauvegarder les données dans le dossier embedded_data
    embedded_folder_path = f"data/embedded_data/{embedded_folder}"
    if not os.path.exists(embedded_folder_path):
        os.makedirs(embedded_folder_path)

    output_path = f"{embedded_folder_path}/{raw_folder}.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Embedded data saved at {output_path}")


def get_bertweet_embeddings(tweet, tokenizer, model):
    # Tokenisation
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True)
    # Passage dans le modèle
    with torch.no_grad():
        outputs = model(**inputs)
    # Utilisation de la dernière couche cachée moyenne comme embedding
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()