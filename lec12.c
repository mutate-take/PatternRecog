#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <string.h>
#include <limits.h>

#define TRUE 1
#define FALSE 0

struct pgm
{
    char magic[3];
    int width;
    int height;
    int max;
    int *image;
};

struct areaFeatures
{
    int area;
    int total_x;
    int total_y;
    int centerOfGravity_x;
    int centerOfGravity_y;
};

struct index_2d
{
    int x;
    int y;
};

struct cluster_attribute
{
    int center;
    int preCenter;
    int pattern_num;
    int feature_sum;
};

struct pattern_info
{
    int cluster_number;
    struct areaFeatures *features;
};

enum direction
{
    northwest,
    north,
    northeast,
    west,
    centor,
    east,
    southwest,
    south,
    southeast
};

int changeIndex_2dimTo1dim(int x, int y, int width)
{
    return y * width + x;
}

struct pgm read_pgm(char *filename)
{
    struct pgm pgm;
    FILE *infp;
    infp = fopen(filename, "r");
    if (infp == NULL)
    {
        fprintf(stderr, "%s:ファイルが開けません", filename);
        fclose(infp);
        exit(1);
    }

    // ヘッダ読み込み
    fscanf(infp, "%s", pgm.magic);
    fscanf(infp, "%d %d", &pgm.width, &pgm.height);
    fscanf(infp, "%d", &pgm.max);

    //一次元配列を動的確保
    pgm.image = (int *)calloc(pgm.height * pgm.width, sizeof(int));
    if (pgm.image == NULL)
    {
        fprintf(stderr, "メモリ確保に失敗しました。");
        fclose(infp);
        exit(1);
    }

    // 画像ファイル読み込み
    for (int i = 0; i < pgm.height; i++)
    {
        for (int j = 0; j < pgm.width; j++)
        {
            //配列に代入
            int current_index = i * pgm.width + j;
            fscanf(infp, "%d", &pgm.image[current_index]);
        }
    }
    fclose(infp);
    return pgm;
}

void out_pgm(char *filename, struct pgm pgm)
{
    FILE *outfp;
    outfp = fopen(filename, "w");

    fprintf(outfp, "%s\n", pgm.magic);
    fprintf(outfp, "%d %d\n", pgm.width, pgm.height);
    fprintf(outfp, "%d\n", pgm.max);

    for (int i = 0; i < pgm.height; i++)
    {
        for (int j = 0; j < pgm.width; j++)
        {
            int current_index = i * pgm.width + j;
            if (current_index % 70 == 0 && current_index != 0)
            {
                fprintf(outfp, "%d\n", pgm.image[current_index]);
            }
            else
            {
                fprintf(outfp, "%d ", pgm.image[current_index]);
            }
        }
    }
    fprintf(outfp, "\n");
    fclose(outfp);
}

int search_max_number(struct pgm pgm)
{
    int max_pic_num = 0;
    for (int i = 0; i < pgm.height; i++)
    {
        for (int j = 0; j < pgm.width; j++)
        {
            int current_index = i * pgm.width + j;
            int current_pic_num = pgm.image[current_index];
            if (max_pic_num < current_pic_num)
            {
                max_pic_num = current_pic_num;
            }
        }
    }
    return max_pic_num;
}

int search_min_number(struct pgm pgm)
{
    int min_pic_num = pgm.max;
    for (int i = 0; i < pgm.height; i++)
    {
        for (int j = 0; j < pgm.width; j++)
        {
            int current_index = i * pgm.width + j;
            int current_pic_num = pgm.image[current_index];
            if (min_pic_num > current_pic_num)
            {
                min_pic_num = current_pic_num;
            }
        }
    }
    return min_pic_num;
}

struct pgm up_contrust(struct pgm pgm)
{
    int max_pic_num = search_max_number(pgm);
    int min_pic_num = search_min_number(pgm);
    // すべてのピクセルの値が同じ　すなわちコントラストが上がらないとき
    if (max_pic_num == min_pic_num)
    {
        printf("コントラストを上げる必要がありません。");
        return pgm;
    }

    for (int i = 0; i < pgm.height; i++)
    {
        for (int j = 0; j < pgm.width; j++)
        {
            int current_index = i * pgm.width + j;
            int current_pic_num = pgm.image[current_index];
            // コントラスト変調式
            double numerator = current_pic_num - min_pic_num;
            // 分母は0でない
            double denominator = max_pic_num - min_pic_num;
            double quotient = numerator / denominator;
            int new_pic_num = (int)(quotient * pgm.max);
            pgm.image[current_index] = new_pic_num;
        }
    }
    return pgm;
}

struct pgm inversion(struct pgm pgm)
{
    for (int i = 0; i < pgm.height; i++)
    {
        for (int j = 0; j < pgm.width; j++)
        {
            int current_index = i * pgm.width + j;
            int current_pic_num = pgm.image[current_index];
            pgm.image[current_index] = pgm.max - current_pic_num;
        }
    }
    return pgm;
}

struct pgm smoothing_by_moving_average(struct pgm pgm)
{
    struct pgm pgm_out;
    // copy
    pgm_out = pgm;

    //別の一次元配列を動的確保
    pgm_out.image = (int *)calloc(pgm.height * pgm.width, sizeof(int));
    if (pgm_out.image == NULL)
    {
        fprintf(stderr, "メモリ確保に失敗しました。");
        exit(1);
    }

    for (int i = 1; i < pgm.height - 1; i++)
    {
        for (int j = 1; j < pgm.width - 1; j++)
        {
            int current_index = i * pgm.width + j;

            int topleft_index_from_current = (i - 1) * pgm.width + (j - 1);
            int left_index_from_current = current_index - 1;
            int bottomleft_index_from_current = (i + 1) * pgm.width + (j - 1);

            // 9マスの平均値
            pgm_out.image[current_index] = (pgm.image[topleft_index_from_current] + pgm.image[topleft_index_from_current + 1] + pgm.image[topleft_index_from_current + 2] + pgm.image[left_index_from_current] + pgm.image[current_index] + pgm.image[left_index_from_current + 2] + pgm.image[bottomleft_index_from_current] + pgm.image[bottomleft_index_from_current + 1] + pgm.image[bottomleft_index_from_current]) / 9;
        }
    }
    return pgm_out;
}

void bubble_sort(int *num, int numlen)
{
    for (int i = 0; i < numlen - 1; i++)
    {
        for (int j = 0; j < numlen - 1 - i; j++)
        {
            if (num[j] > num[j + 1])
            {
                int tmp = num[j];
                num[j] = num[j + 1];
                num[j + 1] = tmp;
            }
        }
    }
}

int get_median(int *num, int numlen)
{
    if (numlen == 0)
    {
        fprintf(stderr, "numlen == 0の中央値は計算できません。");
        exit(1);
    }
    bubble_sort(num, numlen);

    int index = numlen / 2;

    if ((numlen % 2) == 0)
    {
        int median = (num[index - 1] + num[index]) / 2;
        return median;
    }
    else
    {
        return num[index];
    }
}

struct pgm median_filter(struct pgm pgm)
{
    struct pgm pgm_out;
    // copy
    pgm_out = pgm;

    //別の一次元配列を動的確保
    pgm_out.image = (int *)calloc(pgm.height * pgm.width, sizeof(int));
    if (pgm_out.image == NULL)
    {
        fprintf(stderr, "メモリ確保に失敗しました。");
        exit(1);
    }

    for (int i = 1; i < pgm.height - 1; i++)
    {
        for (int j = 1; j < pgm.width - 1; j++)
        {
            int current_index = i * pgm.width + j;

            int topleft_index_from_current = (i - 1) * pgm.width + (j - 1);
            int left_index_from_current = current_index - 1;
            int bottomleft_index_from_current = (i + 1) * pgm.width + (j - 1);

            // 9マスの値を保存
            int values_around_current[9];
            for (int i = 0; i < 3; i++)
            {
                values_around_current[i] = pgm.image[topleft_index_from_current + i];
            }

            for (int i = 0; i < 3; i++)
            {
                values_around_current[i + 3] = pgm.image[left_index_from_current + i];
            }

            for (int i = 0; i < 3; i++)
            {
                values_around_current[i + 6] = pgm.image[bottomleft_index_from_current + i];
            }
            pgm_out.image[current_index] = get_median(values_around_current, sizeof(values_around_current) / sizeof(int));
        }
    }
    return pgm_out;
}

struct pgm divblock(struct pgm pgm, int bsize)
{

    for (int i = 0; i < pgm.height - bsize; i += bsize)
    {
        for (int j = 0; j <= pgm.width - bsize; j += bsize)
        {
            int block_sum = 0;
            int block_mean = 0;
            // ブロックの全画素の平均を求める
            for (int ii = i; ii < i + bsize; ii++)
            {
                for (int jj = j; jj < j + bsize; jj++)
                {
                    int current_index = ii * pgm.width + jj;
                    block_sum += pgm.image[current_index];
                }
            }
            block_mean = block_sum / (bsize * bsize);

            // 平均値をすべての要素への書き込み
            for (int ii = i; ii < i + bsize; ii++)
            {
                for (int jj = j; jj < j + bsize; jj++)
                {
                    int current_index = ii * pgm.width + jj;
                    pgm.image[current_index] = block_mean;
                }
            }
        }
    }
    return pgm;
}

struct pgm expand_x2(struct pgm pgm)
{
    struct pgm pgm_out;
    pgm_out = pgm;

    // pgmファイルの縦横二倍に設定し、配列を動的確保
    pgm_out.width = pgm.width * 2;
    pgm_out.height = pgm.height * 2;

    pgm_out.image = (int *)calloc(pgm_out.height * pgm_out.width, sizeof(int));
    if (pgm_out.image == NULL)
    {
        fprintf(stderr, "メモリ確保に失敗しました。");
        exit(1);
    }

    for (int i = 0; i < pgm_out.height; i++)
    {
        for (int j = 0; j < pgm_out.width; j++)
        {
            int pgm_i = i / 2;
            int pgm_j = j / 2;
            int pgm_current_index = pgm_i * pgm.width + pgm_j;
            int pgm_out_current_index = i * pgm_out.width + j;
            pgm_out.image[pgm_out_current_index] = pgm.image[pgm_current_index];
        }
    }
    return pgm_out;
}

struct pgm expand_xn_by_linear_interpolation(struct pgm pgm, double expandrate_hori, double expandrate_ver)
{
    struct pgm pgm_out;
    pgm_out = pgm;

    // pgmファイルの縦横n　m倍に設定し、配列を動的確保
    pgm_out.width = pgm.width * expandrate_hori;
    pgm_out.height = pgm.height * expandrate_ver;

    pgm_out.image = (int *)calloc(pgm_out.height * pgm_out.width, sizeof(int));
    if (pgm_out.image == NULL)
    {
        fprintf(stderr, "メモリ確保に失敗しました。");
        exit(1);
    }

    for (int i = 0; i < pgm_out.height; i++)
    {
        for (int j = 0; j < pgm_out.width; j++)
        {
            double pgm_i = i / expandrate_ver;
            double pgm_j = j / expandrate_hori;

            if (0 <= pgm_j && pgm_j < pgm.width - 1)
            {
                if (0 <= pgm_i && pgm_i < pgm.height - 1)
                {
                    // 線形補間法
                    int u = pgm_j;
                    int v = pgm_i;
                    double arufa = pgm_j - u;
                    double beta = pgm_i - v;

                    int pgm_current_index = changeIndex_2dimTo1dim(pgm_j, pgm_i, pgm.width);

                    int computation = pgm.image[changeIndex_2dimTo1dim(u, v, pgm.width)] * (1 - arufa) * (1 - beta) + pgm.image[changeIndex_2dimTo1dim(u + 1, v, pgm.width)] * arufa * (1 - beta) + pgm.image[changeIndex_2dimTo1dim(u, v + 1, pgm.width)] * (1 - arufa) * beta + pgm.image[changeIndex_2dimTo1dim(u + 1, v + 1, pgm.width)] * arufa * beta;

                    int pgm_out_current_index = changeIndex_2dimTo1dim(j, i, pgm_out.width);
                    pgm_out.image[pgm_out_current_index] = computation;
                }
            }
        }
    }
    return pgm_out;
}

double degreeMeasure2Radian(double degree)
{
    double rad = 0;
    rad = degree * M_PI / 180;
    return rad;
}

// 行列(二次元配列)の添字番号とx,y軸の向きは同じ
// 水平方向 x 垂直方向 y
struct pgm rotation_from_origin_by_degreeMeasure(struct pgm pgm, double angle)
{
    struct pgm pgm_out;
    pgm_out = pgm;
    pgm_out.image = (int *)calloc(pgm_out.height * pgm_out.width, sizeof(int));
    if (pgm_out.image == NULL)
    {
        fprintf(stderr, "メモリ確保に失敗しました。");
        exit(1);
    }

    double rad = degreeMeasure2Radian(angle);
    // 処理軽減のためループの外で確保
    double cosTHETA = cos(rad);
    double sinTHETA = sin(rad);

    for (int y = 0; y < pgm_out.height; y++)
    {
        for (int x = 0; x < pgm_out.width; x++)
        {

            double x0 = cosTHETA * x + sinTHETA * y;
            double y0 = -sinTHETA * x + cosTHETA * y;

            if (0 <= x0 && x0 < pgm_out.width - 1 && 0 <= y0 && y0 < pgm_out.height - 1)
            {
                int u = x0;
                int v = y0;
                double arufa = x0 - u;
                double beta = y0 - v;

                int computation = pgm.image[changeIndex_2dimTo1dim(u, v, pgm.width)] * (1 - arufa) * (1 - beta) + pgm.image[changeIndex_2dimTo1dim(u + 1, v, pgm.width)] * arufa * (1 - beta) + pgm.image[changeIndex_2dimTo1dim(u, v + 1, pgm.width)] * (1 - arufa) * beta + pgm.image[changeIndex_2dimTo1dim(u + 1, v + 1, pgm.width)] * arufa * beta;

                int pgm_out_current_index = changeIndex_2dimTo1dim(x, y, pgm_out.width);
                pgm_out.image[pgm_out_current_index] = computation;
            }
        }
    }
    return pgm_out;
}

// 行列(二次元配列)の添字番号とx,y軸の向きは同じ
// 水平方向 x 垂直方向 y
struct pgm rotation_from_anypoint_by_degreeMeasure(struct pgm pgm, double angle, int central_x, int central_y)
{
    struct pgm pgm_out;
    pgm_out = pgm;
    pgm_out.image = (int *)calloc(pgm_out.height * pgm_out.width, sizeof(int));
    if (pgm_out.image == NULL)
    {
        fprintf(stderr, "メモリ確保に失敗しました。");
        exit(1);
    }

    double rad = degreeMeasure2Radian(angle);
    // 処理軽減のためループの外で確保
    double cosTHETA = cos(rad);
    double sinTHETA = sin(rad);

    for (int y = 0; y < pgm_out.height; y++)
    {
        for (int x = 0; x < pgm_out.width; x++)
        {

            double x0 = cosTHETA * (x - central_x) + sinTHETA * (y - central_y) + central_x;
            double y0 = -sinTHETA * (x - central_x) + cosTHETA * (y - central_y) + central_y;

            if (0 <= x0 && x0 < pgm_out.width - 1 && 0 <= y0 && y0 < pgm_out.height - 1)
            {
                int u = x0;
                int v = y0;
                double arufa = x0 - u;
                double beta = y0 - v;

                int computation = pgm.image[changeIndex_2dimTo1dim(u, v, pgm.width)] * (1 - arufa) * (1 - beta) + pgm.image[changeIndex_2dimTo1dim(u + 1, v, pgm.width)] * arufa * (1 - beta) + pgm.image[changeIndex_2dimTo1dim(u, v + 1, pgm.width)] * (1 - arufa) * beta + pgm.image[changeIndex_2dimTo1dim(u + 1, v + 1, pgm.width)] * arufa * beta;

                int pgm_out_current_index = changeIndex_2dimTo1dim(x, y, pgm_out.width);
                pgm_out.image[pgm_out_current_index] = computation;
            }
        }
    }
    return pgm_out;
}

// 行列(二次元配列)の添字番号とx,y軸の向きは同じ
// 水平方向 x 垂直方向 y
struct pgm affine_transformation(struct pgm pgm, double a, double b, double c, double d, double e, double f)
{
    struct pgm pgm_out;
    pgm_out = pgm;
    pgm_out.image = (int *)calloc(pgm_out.height * pgm_out.width, sizeof(int));
    if (pgm_out.image == NULL)
    {
        fprintf(stderr, "メモリ確保に失敗しました。");
        exit(1);
    }

    for (int y = 0; y < pgm_out.height; y++)
    {
        for (int x = 0; x < pgm_out.width; x++)
        {

            //逆アフィン変換の式
            double coefficient = 1 / (a * e - b * d);
            double x0 = coefficient * (e * x - b * y) + coefficient * (-e * c + b * f);
            double y0 = coefficient * (-d * x + a * y) + coefficient * (c * d - a * f);

            if (0 <= x0 && x0 < pgm_out.width - 1 && 0 <= y0 && y0 < pgm_out.height - 1)
            {
                int u = x0;
                int v = y0;
                double arufa = x0 - u;
                double beta = y0 - v;

                int computation = pgm.image[changeIndex_2dimTo1dim(u, v, pgm.width)] * (1 - arufa) * (1 - beta) + pgm.image[changeIndex_2dimTo1dim(u + 1, v, pgm.width)] * arufa * (1 - beta) + pgm.image[changeIndex_2dimTo1dim(u, v + 1, pgm.width)] * (1 - arufa) * beta + pgm.image[changeIndex_2dimTo1dim(u + 1, v + 1, pgm.width)] * arufa * beta;

                int pgm_out_current_index = changeIndex_2dimTo1dim(x, y, pgm_out.width);
                pgm_out.image[pgm_out_current_index] = computation;
            }
        }
    }
    return pgm_out;
}

struct pgm binarization_by_threshold(struct pgm pgm, int threshold)
{
    for (int i = 0; i < pgm.height; i++)
    {
        for (int j = 0; j < pgm.width; j++)
        {
            int current_index = changeIndex_2dimTo1dim(j, i, pgm.width);
            int current_pic_num = pgm.image[current_index];
            if (current_pic_num > threshold)
            {
                pgm.image[current_index] = pgm.max;
            }
            else
            {
                pgm.image[current_index] = 0;
            }
        }
    }
    return pgm;
}

int *get_histgramByPicValue(struct pgm pgm)
{

    //一次元配列を動的確保
    int *histgram = (int *)calloc((pgm.max + 1), sizeof(int));
    if (histgram == NULL)
    {
        fprintf(stderr, "メモリ確保に失敗しました。");
        exit(1);
    }

    for (int i = 0; i <= pgm.max; i++)
    {
        histgram[i] = 0;
    }

    for (int y = 0; y < pgm.height; y++)
    {
        for (int x = 0; x < pgm.width; x++)
        {
            int currentIndex = changeIndex_2dimTo1dim(x, y, pgm.width);
            int pic = pgm.image[currentIndex];
            histgram[pic]++;
        }
    }

    return histgram;
}

// parameter : 母数
double getProbablyOfOccurrence_inClassK(int end_pic, int pixel_number, int *histgram)
{
    double sum_P_i = 0;
    // クラスの母数
    double P_i = 0;

    //最後にまとめてpixel_numberで割る
    for (int i = 0; i <= end_pic; i++)
    {
        P_i = histgram[i] / (double)pixel_number;
        sum_P_i += P_i;
    }
    return sum_P_i;
}

double getMean_by_histgram(int end_pic, int pixel_number, int *histgram)
{
    double mu = 0;

    for (int i = 0; i <= end_pic; i++)
    {
        double iP_i = i * ((double)histgram[i] / (double)pixel_number);
        mu += iP_i;
    }
    return mu;
}

double getVarianceBetweenClasses(double threshold, double mu_t, int pixel_number, int *histgram)
{
    double k_variance = 0;
    // しきい値 thresholdごとに変化する値
    double omega_k = getProbablyOfOccurrence_inClassK(threshold, pixel_number, histgram);
    double mu_k = getMean_by_histgram(threshold, pixel_number, histgram);

    double numerator = pow((mu_t * omega_k - mu_k), 2);
    double denominator = omega_k * (1 - omega_k);
    k_variance = numerator / denominator;
    return k_variance;
}

int getBestThresholdByDiscriminantAnalysisMethod(struct pgm pgm)
{
    // しきい値 threshold が変化しても普遍の値
    int *histgram = get_histgramByPicValue(pgm);

    int pixel_num = pgm.width * pgm.height;

    // 負の極限の意味
    double max_variance = -1000000000000000;
    int bestThreshold = -1;
    double mu_t = getMean_by_histgram(pgm.max, pixel_num, histgram);
    for (int k = 0; k <= pgm.max - 1; k++)
    {
        double k_variance = getVarianceBetweenClasses(k, mu_t, pixel_num, histgram);
        if (max_variance < k_variance)
        {
            max_variance = k_variance;
            bestThreshold = k;
        }
    }
    free(histgram);
    return bestThreshold;
}

struct pgm BinaryzationByDiscriminantAnalysisMethod(struct pgm pgm)
{
    int threshold = getBestThresholdByDiscriminantAnalysisMethod(pgm);
    printf("threshold : %d\n", threshold);
    // pgm = binarization_by_threshold(pgm, threshold);
    pgm = binarization_by_threshold(pgm, threshold);
    return pgm;
}

int checkOutOfIndex(int x, int y, int width, int height)
{
    // 配列外 -1 配列内 0
    if (x <= -1)
    {
        return TRUE;
    }
    if (x >= width)
    {
        return TRUE;
    }
    if (y <= -1)
    {
        return TRUE;
    }
    if (y >= height)
    {
        return TRUE;
    }
    return FALSE;
}

void labeling_reach_1groupBy8neighbor(int x, int y, int label, struct pgm pgm)
{
    // 配列外のindexを指すかのチェック後に再帰
    int current_index = changeIndex_2dimTo1dim(x, y, pgm.width);
    pgm.image[current_index] = label;
    //8近傍を探索

    // 1.基準マスの北西
    if (checkOutOfIndex(x - 1, y - 1, pgm.width, pgm.height) == FALSE)
    {
        int current_index = changeIndex_2dimTo1dim(x - 1, y - 1, pgm.width);
        if (pgm.image[current_index] == pgm.max)
        {
            labeling_reach_1groupBy8neighbor(x - 1, y - 1, label, pgm);
        }
    }

    // 2.基準マスの北へ
    if (checkOutOfIndex(x, y - 1, pgm.width, pgm.height) == FALSE)
    {
        int current_index = changeIndex_2dimTo1dim(x, y - 1, pgm.width);
        if (pgm.image[current_index] == pgm.max)
        {
            labeling_reach_1groupBy8neighbor(x, y - 1, label, pgm);
        }
    }

    // 3.基準マスの北東へ
    if (checkOutOfIndex(x + 1, y - 1, pgm.width, pgm.height) == FALSE)
    {
        int current_index = changeIndex_2dimTo1dim(x + 1, y - 1, pgm.width);
        if (pgm.image[current_index] == pgm.max)
        {
            labeling_reach_1groupBy8neighbor(x + 1, y - 1, label, pgm);
        }
    }

    // 4.基準マスの東へ
    if (checkOutOfIndex(x + 1, y, pgm.width, pgm.height) == FALSE)
    {
        int current_index = changeIndex_2dimTo1dim(x + 1, y, pgm.width);
        if (pgm.image[current_index] == pgm.max)
        {
            labeling_reach_1groupBy8neighbor(x + 1, y, label, pgm);
        }
    }

    // 5.基準マスの南東へ
    if (checkOutOfIndex(x + 1, y + 1, pgm.width, pgm.height) == FALSE)
    {
        int current_index = changeIndex_2dimTo1dim(x + 1, y + 1, pgm.width);
        if (pgm.image[current_index] == pgm.max)
        {
            labeling_reach_1groupBy8neighbor(x + 1, y + 1, label, pgm);
        }
    }

    // 6.基準マスの南へ
    if (checkOutOfIndex(x, y + 1, pgm.width, pgm.height) == FALSE)
    {
        int current_index = changeIndex_2dimTo1dim(x, y + 1, pgm.width);
        if (pgm.image[current_index] == pgm.max)
        {
            labeling_reach_1groupBy8neighbor(x, y + 1, label, pgm);
        }
    }

    // 7.基準マスの南西へ
    if (checkOutOfIndex(x - 1, y + 1, pgm.width, pgm.height) == FALSE)
    {
        int current_index = changeIndex_2dimTo1dim(x - 1, y + 1, pgm.width);
        if (pgm.image[current_index] == pgm.max)
        {
            labeling_reach_1groupBy8neighbor(x - 1, y + 1, label, pgm);
        }
    }

    // 8.基準マスの西へ
    if (checkOutOfIndex(x - 1, y, pgm.width, pgm.height) == FALSE)
    {
        int current_index = changeIndex_2dimTo1dim(x - 1, y, pgm.width);
        if (pgm.image[current_index] == pgm.max)
        {
            labeling_reach_1groupBy8neighbor(x - 1, y, label, pgm);
        }
    }

    return;
}

struct pgm labeling(struct pgm pgm)
{
    int label = 1;
    for (int y = 0; y < pgm.height; y++)
    {
        for (int x = 0; x < pgm.width; x++)
        {
            int current_index = changeIndex_2dimTo1dim(x, y, pgm.width);
            int pic_val = pgm.image[current_index];

            if (pic_val == pgm.max)
            {
                labeling_reach_1groupBy8neighbor(x, y, label, pgm);
                label++;
            }
        }
    }
    return pgm;
}

int computeMaxLabel(struct pgm pgm)
{
    int maxlabel = -1;
    for (int y = 0; y < pgm.height; y++)
    {
        for (int x = 0; x < pgm.width; x++)
        {
            int current_index = changeIndex_2dimTo1dim(x, y, pgm.width);
            int pic_num = pgm.image[current_index];
            if (pic_num > maxlabel)
            {
                maxlabel = pic_num;
            }
            // printf("%d", pic_num);
        }
    }
    return maxlabel;
}

struct areaFeatures *computeFeatures_EachLabel(struct pgm pgm, int maxlabel)
{
    struct areaFeatures *features;
    features = calloc(maxlabel + 1, sizeof(struct areaFeatures));

    if (features == NULL)
    {
        fprintf(stderr, "メモリの確保に失敗しました。\n");
        exit(1);
    }
    // 初期化
    for (int i = 0; i <= maxlabel; i++)
    {
        features[i].area = 0;
        features[i].total_x = 0;
        features[i].total_y = 0;
        features[i].centerOfGravity_x = 0;
        features[i].centerOfGravity_y = 0;
    }

    // ラベルごとの面積、x座標の合計、y座標の合計を計算
    for (int y = 0; y < pgm.height; y++)
    {
        for (int x = 0; x < pgm.width; x++)
        {
            int current_index = changeIndex_2dimTo1dim(x, y, pgm.width);
            int label = pgm.image[current_index];
            if (label != 0)
            {
                features[label].area += 1;
                features[label].total_x += x;
                features[label].total_y += y;
            }
        }
    }

    for (int i = 1; i <= maxlabel; i++)
    {
        features[i].centerOfGravity_x = features[i].total_x / features[i].area;
        features[i].centerOfGravity_y = features[i].total_y / features[i].area;
    }
    return features;
}

void printFeatures(struct areaFeatures *features)
{
    int size = malloc_usable_size(features) / sizeof(struct areaFeatures);
    printf("%d", size);
    printf("label number\tarea\txwt\tywt\n");
    for (int i = 1; i < size; i++)
    {
        printf("\t%d\t%d\t%d\t%d\n", i, features[i].area, features[i].centerOfGravity_x, features[i].centerOfGravity_y);
    }
}

struct pgm ExtractAreaOfLabelNumber(struct pgm original_pgm, struct pgm rabeled_pgm, int label)
{
    for (int y = 0; y < rabeled_pgm.height; y++)
    {
        for (int x = 0; x < rabeled_pgm.width; x++)
        {
            int current_index = changeIndex_2dimTo1dim(x, y, rabeled_pgm.width);
            int label_or_0 = rabeled_pgm.image[current_index];
            if (label_or_0 != label)
            {
                original_pgm.image[current_index] = 0;
            }
        }
    }

    return original_pgm;
}

int getLabelOfMaxArea(struct areaFeatures *features)
{
    int length = malloc_usable_size(features) / sizeof(struct areaFeatures);
    int maxarea = -1;
    int label = -1;
    for (int i = 0; i < length; i++)
    {
        if (maxarea < features[i].area)
        {
            maxarea = features[i].area;
            label = i;
        }
    }
    return label;
}

struct pgm duplicate_pgm(struct pgm pgm_original)
{
    struct pgm pgm_copy;
    pgm_copy = pgm_original;
    pgm_copy.image = (int *)calloc(pgm_original.height * pgm_original.width, sizeof(int));

    memcpy(pgm_copy.image, pgm_original.image, malloc_usable_size(pgm_original.image));
    return pgm_copy;
}

int getSum_AbsoluteValueOfDifference(struct pgm template, struct pgm pgm_source, int sharedLeftEndIndex_x, int sharedLeftEndIndex_y, int currrent_MinDistance)
{

    int sum_abso_dif = 0;
    // 配列のoutOfIndexチェックは呼び出し元で防いでいる
    for (int y = sharedLeftEndIndex_y; y < sharedLeftEndIndex_y + template.height; y++)
    {
        for (int x = sharedLeftEndIndex_x; x < sharedLeftEndIndex_x + template.width; x++)
        {
            int current_index_template = changeIndex_2dimTo1dim(x - sharedLeftEndIndex_x, y - sharedLeftEndIndex_y, template.width);
            int current_index_source = changeIndex_2dimTo1dim(x, y, pgm_source.width);
            int picValueOfTemplate = template.image[current_index_template];
            int picValueOfSource = pgm_source.image[current_index_source];

            int absValueOfDif = abs(picValueOfSource - picValueOfTemplate);
            sum_abso_dif += absValueOfDif;
            if (sum_abso_dif > currrent_MinDistance)
            {
                return -1;
            }
        }
    }
    return sum_abso_dif;
}

struct index_2d getIndexOFTemplateMatchingByDistance(struct pgm template, struct pgm pgm_source)
{
    int minD = INT_MAX;
    struct index_2d upper_left_index;
    upper_left_index.x = 0;
    upper_left_index.y = 0;
    for (int y = 0; y < pgm_source.height - template.height; y++)
    {
        for (int x = 0; x < pgm_source.width - template.width; x++)
        {
            int distance_or_SuspendState = getSum_AbsoluteValueOfDifference(template, pgm_source, x, y, minD);
            if (distance_or_SuspendState == -1)
            {
                // 計算の中断
                ;
            }
            else if (distance_or_SuspendState < minD)
            {
                minD = distance_or_SuspendState;
                upper_left_index.x = x;
                upper_left_index.y = y;
            }
        }
    }
    printf("minD : %d\n", minD);
    return upper_left_index;
}

struct pgm surroundMatchingTemplate(struct pgm pgm, struct index_2d index_info, int template_width, int template_height)
{
    for (int y = index_info.y; y < index_info.y + template_height; y++)
    {
        pgm.image[changeIndex_2dimTo1dim(index_info.x, y, pgm.width)] = pgm.max;
        pgm.image[changeIndex_2dimTo1dim(index_info.x + template_width, y, pgm.width)] = pgm.max;
    }
    for (int x = index_info.x; x < index_info.x + template_width; x++)
    {
        pgm.image[changeIndex_2dimTo1dim(x, index_info.y, pgm.width)] = pgm.max;
        pgm.image[changeIndex_2dimTo1dim(x, index_info.y + template_height, pgm.width)] = pgm.max;
    }

    return pgm;
}

double getDegreeOfSimilarity(struct pgm template, struct pgm pgm_source, int sharedLeftEndIndex_x, int sharedLeftEndIndex_y, double currrent_Max_S)
{

    double similarity = 0;
    int IT = 0;
    int TT = 0;
    int II = 0;
    // 配列のoutOfIndexチェックは呼び出し元で防いでいる
    for (int y = sharedLeftEndIndex_y; y < sharedLeftEndIndex_y + template.height; y++)
    {
        for (int x = sharedLeftEndIndex_x; x < sharedLeftEndIndex_x + template.width; x++)
        {
            int current_index_template = changeIndex_2dimTo1dim(x - sharedLeftEndIndex_x, y - sharedLeftEndIndex_y, template.width);
            int current_index_source = changeIndex_2dimTo1dim(x, y, pgm_source.width);
            int picValueOfTemplate = template.image[current_index_template];
            int picValueOfSource = pgm_source.image[current_index_source];

            IT += picValueOfSource * picValueOfTemplate;
            II += picValueOfSource * picValueOfSource;
            TT += picValueOfTemplate * picValueOfTemplate;
        }
    }
    similarity = IT / (sqrt(II) * sqrt(TT));

    return similarity;
}

struct index_2d getIndexOFTemplateMatchingBySimilarity(struct pgm template, struct pgm pgm_source)
{
    double maxS = 0;
    struct index_2d upper_left_index;
    upper_left_index.x = 0;
    upper_left_index.y = 0;
    for (int y = 0; y < pgm_source.height - template.height; y++)
    {
        for (int x = 0; x < pgm_source.width - template.width; x++)
        {
            double similarity = getDegreeOfSimilarity(template, pgm_source, x, y, maxS);
            if (similarity > maxS)
            {
                maxS = similarity;
                upper_left_index.x = x;
                upper_left_index.y = y;
            }
        }
    }
    printf("maxS : %f\n", maxS);
    return upper_left_index;
}

struct pattern_info *initializePatternInfo(struct areaFeatures *areaFeatures)
{
    struct pattern_info *pattern_infos;
    int pattern_num = malloc_usable_size(areaFeatures) / sizeof(struct areaFeatures);
    pattern_infos = calloc(pattern_num, sizeof(struct pattern_info));
    if (pattern_infos == NULL)
    {
        fprintf(stderr, "メモリ確保に失敗しました。");
        exit(1);
    }

    for (int i = 0; i < pattern_num; i++)
    {
        pattern_infos[i].cluster_number = 0;
        pattern_infos[i].features = areaFeatures;
    }
    return pattern_infos;
}

struct cluster_attribute *initializeClusterAttribute(int cluster_num, struct pattern_info *pattern_infos)
{
    struct cluster_attribute *cluster_infos;
    cluster_infos = (struct cluster_attribute *)calloc(cluster_num + 1, sizeof(struct cluster_attribute));
    if (cluster_infos == NULL)
    {
        fprintf(stderr, "メモリ確保に失敗しました。");
        exit(1);
    }

    for (int i = 0; i <= cluster_num; i++)
    {
        cluster_infos[i].center = pattern_infos[i].features[i].area;
        cluster_infos[i].preCenter = 0;
        cluster_infos[i].pattern_num = 1;
        cluster_infos[i].feature_sum = pattern_infos[i].features[i].area;
    }
    return cluster_infos;
}

void setPatternNumAndPatternSumOfEachCluster(struct pattern_info *pattern_infos, struct cluster_attribute *cluster_infos)
{
    int cluster_num = malloc_usable_size(cluster_infos) / sizeof(struct cluster_attribute) - 1;
    int pattern_num = malloc_usable_size(pattern_infos) / sizeof(struct pattern_info) - 1;

    for (int p = 1; p <= pattern_num; p++)
    {
        int clusterNumber = pattern_infos[p].cluster_number;
        cluster_infos[clusterNumber].feature_sum += pattern_infos[p].features[p].area;
        cluster_infos[clusterNumber].pattern_num += 1;
    }
}

void updateCenterOfCluster(struct cluster_attribute *cluster_infos)
{
    int cluster_num = malloc_usable_size(cluster_infos) / sizeof(struct cluster_attribute) - 1;
    for (int i = 1; i <= cluster_num; i++)
    {
        cluster_infos[i].preCenter = cluster_infos[i].center;
        int nextcenter = cluster_infos[i].feature_sum / cluster_infos[i].pattern_num;
        cluster_infos[i].center = nextcenter;
    }
}

int computeSumOfDifferencesCenterFromPreCenter(struct cluster_attribute *cluster_infos)
{
    int cluster_num = malloc_usable_size(cluster_infos) / sizeof(struct cluster_attribute) - 1;
    int sum = 0;
    for (int i = 1; i <= cluster_num; i++)
    {
        int dif = abs(cluster_infos[i].center - cluster_infos[i].preCenter);
        sum += dif;
    }
    return sum;
}

void printPattern_infos(struct pattern_info *pattern_infos)
{
    int pattern_num = malloc_usable_size(pattern_infos) / sizeof(struct pattern_info) - 1;
    printf("label number \tarea\tcluster number\n");
    for (int i = 1; i <= pattern_num; i++)
    {
        printf("\t%d\t%d\t%d\n", i, pattern_infos[i].features[i].area, pattern_infos[i].cluster_number);
    }
}

void extract_pgm(char *filename, struct pgm pgm, struct index_2d upper_left_index, int extract_width, int extract_height)
{
    struct pgm subpgm;
    subpgm = pgm;
    subpgm.height = extract_height;
    subpgm.width = extract_width;

    subpgm.image = (int *)calloc(extract_height * extract_width, sizeof(int));
    if (subpgm.image == NULL)
    {
        fprintf(stderr, "メモリ確保に失敗しました。");
        exit(1);
    }

    int subpgm_index = 0;
    // original_arrayから、subarrayを取得
    for (int y = upper_left_index.y; y < upper_left_index.y + extract_height; y++)
    {
        for (int x = upper_left_index.x; x < upper_left_index.x + extract_width; x++)
        {
            int current_index = changeIndex_2dimTo1dim(x, y, pgm.width);
            subpgm.image[subpgm_index] = pgm.image[current_index];
            subpgm_index++;
        }
    }
    out_pgm(filename, subpgm);
}

void generateNextTemplate(struct index_2d upper_left, struct pgm cur_template, struct pgm input_file, char *next_template_filename)
{
    // printf("x : %d y : %d\n", upper_left.x, upper_left.y);
    extract_pgm(next_template_filename, input_file, upper_left, cur_template.width, cur_template.height);
}

void generateMatchingResult(struct index_2d upper_left, struct pgm pgm, struct pgm template, char* outfilename){
    pgm = surroundMatchingTemplate(pgm, upper_left, template.width, template.height);
    out_pgm(outfilename, pgm);

}

void templateMatching(struct pgm first_template, struct pgm *input_files, int input_file_num)
{
    struct pgm cur_template = first_template;
    for (int i = 0; i < input_file_num; i++)
    {
        struct pgm nextTemplate;
        struct index_2d upper_left = getIndexOFTemplateMatchingBySimilarity(cur_template, input_files[i]);

        char nextTemplateName[64];
        char markedFileName[64];
        sprintf(nextTemplateName, "template%d", i + 1);
        sprintf(markedFileName, "marked%d", i + 1);
        generateNextTemplate(upper_left ,cur_template, input_files[i], nextTemplateName);
        generateMatchingResult(upper_left, input_files[i], cur_template, markedFileName);

        cur_template = read_pgm(nextTemplateName);
    }
}

int main(int arc, char **argv)
{
    struct pgm *input_files;
    struct pgm pgm_template;
    int inputfile_num = 0;
    char *infile_name_symbolic_part;
    char *templatefile_name;
    char *outfile_name;

    if (arc != 4)
    {
        fprintf(stderr, "usage : ./'program' templatefile inputfile_num infilename_symbolic_part\n");
        exit(1);
    }

    templatefile_name = argv[1];
    inputfile_num = atoi(argv[2]);
    infile_name_symbolic_part = argv[3];

    input_files = (struct pgm *)calloc(inputfile_num, sizeof(struct pgm));
    if (input_files == NULL)
    {
        fprintf(stderr, "メモリ確保に失敗しました。");
        exit(1);
    }

    // 複数の入力ファイルをメモリ上へ
    for (int i = 0; i < inputfile_num; i++)
    {
        char filename[64];
        sprintf(filename, "%s%d.pgm", infile_name_symbolic_part, i + 1);
        input_files[i] = read_pgm(filename);
    }

    // template画像もメモリ上へ
    pgm_template = read_pgm(templatefile_name);

    templateMatching(pgm_template, input_files, inputfile_num);

    // メモリの解放
    for (int i = 0; i < inputfile_num; i++)
    {
        free(input_files[i].image);
    }
    free(input_files);
    free(pgm_template.image);

    return 0;
}