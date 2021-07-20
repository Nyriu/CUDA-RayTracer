#!/bin/ipython3

import pandas as pd
from plotnine import *
from dplython import *

#times_filename = "1626815008.txt"
times_filename = "1626852881.txt"

#times_dir = "../times/" 
times_dir = "./times/"

times_csv = pd.read_csv(times_dir + times_filename)
times_dpl = DplyFrame(times_csv)

(times_dpl >> head(1))

# -----------------------------------------------------------------------------------
# Per Frame comparison
print("Per Frame comparison")
data_to_show_parz1 = \
        pd.DataFrame(
                (
                    times_dpl >>
                    sift(X.rnd_or_enc == 1) >>
                    select(X.id, X.seed_or_code, X.additional, X.frame_num, X.total_microsec, X.update_time, X.render_time)
                    )
                )
#data_to_show_parz1["time_type"] = "update"



data_to_show = (DplyFrame(
        data_to_show_parz1#.append(
            #data_to_show_parz2.append(
            #    data_to_show_parz3
            #    )
            #)
        ) #>> sift(X.id == 1) ## TODO remove
        )
data_to_show["30_fps_line"] = 1e6/30
data_to_show["30_fps_line_label"] = "30fps"

rename and titles plot and comment
p = \
(
    ggplot(data_to_show) +
    aes(x="frame_num", y="update_time") +
    geom_line(alpha=0.7) +
    facet_wrap("additional") +
    geom_hline(aes(yintercept="30_fps_line"), color="red") +
    geom_text(aes(data_to_show.frame_num.max()+1.5,"30_fps_line",label="30_fps_line_label")) +
    xlab("Frame Num") +
    ylab("Mean Frame Time(µs)") #+
    #guides(fill=guide_legend(title="Time of ")) +
    #theme_minimal() #+
    #theme(axis_text_x = element_text(angle = 45, hjust = 1))
)
p

filename = times_filename.removesuffix(".txt") + "_frame_comparison.png"
print("filename = ", filename)
ggsave(p,filename,
        #device="png",
        #device="pdf",
        #width=330, height=200#, units="px"
        width=12*2, height=5*2, units="cm"
        )



# -----------------------------------------------------------------------------------
# Mean time
print("Mean time")
data_to_show_parz1 = \
        pd.DataFrame(
                (
                    times_dpl >>
                    sift(X.rnd_or_enc == 0) >> # filter "random"
                    sift(X.update_time != 0 and X.render_time != 0) >>
                    #group_by(X.id,X.img_h,X.img_w,X.n_objs,X.n_lights,X.rnd_or_enc,X.seed_or_code,X.total_microsec, X.additional) >>
                    group_by(X.id) >>
                    summarize(mean_time = X.update_time.mean())
                    )
                )
data_to_show_parz1["time_type"] = "update"

data_to_show_parz2 = \
        pd.DataFrame(
                (
                    times_dpl >>
                    sift(X.rnd_or_enc == 0) >> # filter "random"
                    sift(X.update_time != 0 and X.render_time != 0) >>
                    #group_by(X.id,X.img_h,X.img_w,X.n_objs,X.n_lights,X.rnd_or_enc,X.seed_or_code,X.total_microsec, X.additional) >>
                    group_by(X.id) >>
                    summarize(mean_time = X.render_time.mean())
                    )
                )
data_to_show_parz2["time_type"] = "render"

data_to_show_parz3 = \
        pd.DataFrame(
                (
                    times_dpl >>
                    sift(X.rnd_or_enc == 0) >> # filter "random"
                    sift(X.update_time != 0 and X.render_time != 0) >>
                    #group_by(X.id,X.img_h,X.img_w,X.n_objs,X.n_lights,X.rnd_or_enc,X.seed_or_code,X.total_microsec, X.additional) >>
                    group_by(X.id) >>
                    summarize(mean_time = X.total_microsec.mean())
                    )
                )
data_to_show_parz3["time_type"] = "total"

data_to_show = DplyFrame(
        data_to_show_parz1.append(
            data_to_show_parz2.append(
                data_to_show_parz3
                )
            )
        )
data_to_show["30_fps_line"] = 1e6/30
data_to_show["30_fps_line_label"] = "30fps"
data_to_show

p = \
(
    ggplot(data_to_show) +
    aes(x="id", y="mean_time", fill="time_type") +
    geom_bar(stat="identity", width=.5, position="dodge") +
    #aes(x="id", y="mean_frame_update_time") +
    #geom_col() +
    #aes(x="id", y="mean_frame_render_time") +
    #geom_col() +
    #geom_line() +
    #geom_line(aes(x="id", y="mean_frame_update_time")) +
    geom_hline(aes(yintercept="30_fps_line"), color="red") +
    geom_text(aes(data_to_show.id.max()+1.5,"30_fps_line",label="30_fps_line_label")) + #, vjust=1)) +
    scale_x_discrete(name="Scene Num", limits=range(data_to_show.id.max()+2)) +
    ylab("Mean Frame Time(µs)") +
    guides(fill=guide_legend(title="Time of ")) +
    theme_minimal() +
    theme(axis_text_x = element_text(angle = 45, hjust = 1))
)
p

filename = times_filename.removesuffix(".txt") + "_mean_time.png"
print("filename = ", filename)
ggsave(p,filename,
        #device="png",
        #device="pdf",
        #width=330, height=200#, units="px"
        width=12*2, height=5*2, units="cm"
        )



# -----------------------------------------------------------------------------------
# Scene Complexity
print("Scene Complexity")

data_to_show_parz1 = \
pd.DataFrame(
        (
            times_dpl >>
            sift(X.rnd_or_enc == 0) >> # filter "random"
            sift(X.update_time != 0 and X.render_time != 0) >>
            select(X.id,X.img_h,X.n_objs) >>
            rename(img_dim=X.img_h)
            #select(X.id,X.img_h,X.img_w,X.n_objs) #,X.n_lights)
            #group_by(X.id,X.img_h,X.img_w,X.n_objs,X.n_lights,X.rnd_or_enc,X.seed_or_code,X.total_microsec, X.additional) >>
            #summarize(mean_time = X.render_time.mean())
            )
        )
data_to_show_parz1["objs_or_lights"] = "objects"
data_to_show_parz1["w_or_h"] = "height"
data_to_show_parz1

data_to_show_parz2 = \
pd.DataFrame(
        (
            times_dpl >>
            sift(X.rnd_or_enc == 0) >> # filter "random"
            sift(X.update_time != 0 and X.render_time != 0) >>
            select(X.id,X.img_w,X.n_lights) >>
            rename(n_objs=X.n_lights) >>
            rename(img_dim=X.img_w)
            #group_by(X.id,X.img_h,X.img_w,X.n_objs,X.n_lights,X.rnd_or_enc,X.seed_or_code,X.total_microsec, X.additional) >>
            #summarize(mean_time = X.render_time.mean())
            )
        )
data_to_show_parz2["objs_or_lights"] = "lights"
data_to_show_parz2["w_or_h"] = "width"
data_to_show_parz2

data_to_show = DplyFrame(data_to_show_parz1.append(data_to_show_parz2))

p = \
(
    ggplot(data_to_show) +
    aes(x="id", y="n_objs", fill="objs_or_lights") +
    geom_bar(stat="identity", width=.5, position="dodge") +
    scale_x_discrete(name="Scene Num", limits=range(data_to_show.id.max()+1)) +
    ylab("Number") +
    guides(fill=guide_legend(title="")) +
    theme_minimal() +
    theme(axis_text_x  = element_text(angle = 45, hjust = 1))
)
p

filename = times_filename.removesuffix(".txt") + "_scene_complexity.png"
ggsave(p,filename,
        #device="png",
        #device="pdf",
        #width=330, height=200#, units="px"
        width=12*2, height=5*2, units="cm"
        )




