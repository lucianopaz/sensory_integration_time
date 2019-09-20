#!/bin/bash

show_help() {
help="dispatcher.sh [-OPTIONS]

Valid options:
-h: Display help

-v: Pass option --verbose to python fitter.py calls

-g: Pass option --debug to python fitter.py calls, this overrides any -v option passed

-w: Pass option -w to python fitter.py calls, which causes fit files to be overwritten

-e: Must be followed by a space and a string which represents the option's value.
This is used to partially set the method used by fitter.py. It controls
whether to uses slider, disc or slider_disc methods. [Default is slider_disc]

-o: Must be followed by a space and a string which represents the option's value.
This is used to set the optimizer used by fitter.py. [Default is cma]

-m: Must be followed by a space and a string which represents the option's value.
Merit type. Can be ls or nll [Default ls]

-p: Must be followed by a space and a number which represents the option's value. Set protocol number
understood values are 1, 2 and 3, which causes the fits to be done in two successive runs.
In protocols 1 and 2, the merit type supplied by the -m option is respected.
In protocol 3, the merit type is overriten. In protocols 1 and 2,
the first run uses the method='disc' and 'slider' respectively.
The second run uses the method='slider' and 'disc' respectively, but
also loads the fitted parameters from the first run and leaves them fixed.
These two protocols effectively override whatever is passed in the -e option.
In protocol 3, the first run is done with disc_nll (also disc_2alpha_nll)
and the second run is done with slider_ls (also slider_2alpha_ls). For
graphing purposes, the exps and merit type values are taken as
slider and ls respectively
If any other numerical value is found, then the fitting process goes by
normally with only a single run.  [Default is 0]

-s: Must be followed by a space and a string which represents the option's value.
Can be humans or rats. If the value is rats, then protocol is set to 0 and
exps is set to rats, because rats can only perform the discrimination task.
[Default is humans]

-n: Must be followed by a space and a number which represents the option's value.
Set the number of processes to run in parallel for each alpha condition. At
the moment, there are two alpha conditions single_alpha and 2alpha, so the
total number of process that will be put to run in the background for fitting
will be n*2. [Default is 7]

-y: Must be followed by a space and a string which represents the option's value.
Is the python executable, defaults to python.

"
    echo "$help"
}

OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Initialize our own variables:
protocol=0
verbosity=""
override=""
exps="slider_disc"
optimizer="cma"
merit="ls"
display_commands=0
subjects="humans"
nt=7
python="python"

while getopts "hvgwp:e:m:s:n:y:" opt; do
    case "$opt" in
    h|\?)
        show_help
        exit 0
        ;;
    v)
        if [ "$verbosity"!="--debug" ]; then
            verbosity="--verbose"
        fi
        display_commands=1
        ;;
    g)  verbosity="--debug"
        display_commands=1
        ;;
    w)  override="-w"
        ;;
    e)  exps=$OPTARG
        ;;
    m)  merit=$OPTARG
        ;;
    p)  protocol=$OPTARG
        ;;
    s)  subjects=$OPTARG
        ;;
    n)  nt=$OPTARG
        ;;
    y)  python=$OPTARG
        ;;
    esac
done

command -v "$python" >/dev/null 2>&1 || { echo >&2 "The supplied python executable ($python) was not found.  Aborting."; exit 1; }

clean_comm_whitespaces() {
    echo $1 | sed -e 's/\s*$//g' | sed -e 's/^\s*//g' | sed -e 's/\s\{1,\}/ /g' | sed -e's/\\ / /g';
}
experiment="humans"
if [ "$subjects" == "rats" ]; then
    exps="disc";
    experiment="rats";
    protocol=0;
elif [ "$subjects" != "humans" ]; then
    echo "Unknown -s option \"$subjects\". Available values are humans or rats.";
    exit 1;
fi

if [ $protocol -eq 1 ]; then
    exps="slider";
elif [ $protocol -eq 2 ]; then
    exps="disc";
elif [ $protocol -eq 3 ]; then
    exps="slider";
    merit="ls";
fi

children=()
_term() {
    for child in "${children[@]}"; do
        kill -n 9 "$child";
    done
    exit 1
}
export MPLBACKEND="PDF"

#trap _term SIGINT SIGQUIT SIGABRT SIGKILL SIGTERM ERR
trap _term SIGINT SIGQUIT SIGABRT SIGKILL SIGTERM

task=0
echo "Dispatching fitters"
if [ $protocol -eq 4 ]; then
    commands=()
    commands+=("$python fitter.py -e humans --method slider_disc_ls --merge all --save --save_plot_handler $verbosity $override -o $optimizer --suffix _merged && mv figs/fits_humans_slider_disc_ls_merged.pdf figs/merged_humans_slider_disc_single_alpha_ls.pdf")
    commands+=("$python fitter.py -e humans --method slider_disc_2alpha_ls --merge all --save --save_plot_handler $verbosity $override -o $optimizer --suffix _merged && mv figs/fits_humans_slider_disc_2alpha_ls_merged.pdf figs/merged_humans_slider_disc_2alphas_ls.pdf")
    commands+=("$python fitter.py -e humans --method slider_disc_nll --merge all --save --save_plot_handler $verbosity $override -o $optimizer --suffix _merged && mv figs/fits_humans_slider_disc_nll_merged.pdf figs/merged_humans_slider_disc_single_alpha_nll.pdf")
    commands+=("$python fitter.py -e humans --method slider_disc_2alpha_nll --merge all --save --save_plot_handler $verbosity $override -o $optimizer --suffix _merged && mv figs/fits_humans_slider_disc_2alpha_nll_merged.pdf figs/merged_humans_slider_disc_2alphas_nll.pdf")
    commands+=("$python fitter.py -e humans --method slider_ls --merge all --save --save_plot_handler $verbosity $override -o $optimizer --suffix _merged && mv figs/fits_humans_slider_ls_merged.pdf figs/merged_humans_slider_single_alpha_ls.pdf")
    commands+=("$python fitter.py -e humans --method slider_2alpha_ls --merge all --save --save_plot_handler $verbosity $override -o $optimizer --suffix _merged && mv figs/fits_humans_slider_2alpha_ls_merged.pdf figs/merged_humans_slider_2alphas_ls.pdf")
    commands+=("$python fitter.py -e humans --method slider_nll --merge all --save --save_plot_handler $verbosity $override -o $optimizer --suffix _merged && mv figs/fits_humans_slider_nll_merged.pdf figs/merged_humans_slider_single_alpha_nll.pdf")
    commands+=("$python fitter.py -e humans --method slider_2alpha_nll --merge all --save --save_plot_handler $verbosity $override -o $optimizer --suffix _merged && mv figs/fits_humans_slider_2alpha_nll_merged.pdf figs/merged_humans_slider_2alphas_nll.pdf")
    commands+=("$python fitter.py -e humans --method disc_ls --merge all --save --save_plot_handler $verbosity $override -o $optimizer --suffix _merged && mv figs/fits_humans_disc_ls_merged.pdf figs/merged_humans_disc_single_alpha_ls.pdf")
    commands+=("$python fitter.py -e humans --method disc_2alpha_ls --merge all --save --save_plot_handler $verbosity $override -o $optimizer --suffix _merged && mv figs/fits_humans_disc_2alpha_ls_merged.pdf figs/merged_humans_disc_2alphas_ls.pdf")
    commands+=("$python fitter.py -e humans --method disc_nll --merge all --save --save_plot_handler $verbosity $override -o $optimizer --suffix _merged && mv figs/fits_humans_disc_nll_merged.pdf figs/merged_humans_disc_single_alpha_nll.pdf")
    commands+=("$python fitter.py -e humans --method disc_2alpha_nll --merge all --save --save_plot_handler $verbosity $override -o $optimizer --suffix _merged && mv figs/fits_humans_disc_2alpha_nll_merged.pdf figs/merged_humans_disc_2alphas_nll.pdf")
    commands+=("$python fitter.py -e rats --method disc_2alpha_ls --merge all --save --save_plot_handler $verbosity $override -o $optimizer --suffix _merged && mv figs/fits_rats_disc_2alpha_ls_merged.pdf figs/merged_rats_disc_2alphas_ls.pdf")
    commands+=("$python fitter.py -e rats --method disc_2alpha_nll --merge all --save --save_plot_handler $verbosity $override -o $optimizer --suffix _merged && mv figs/fits_rats_disc_2alpha_nll_merged.pdf figs/merged_rats_disc_2alphas_nll.pdf")
    for command in "${commands[@]}"; do
        if [ $display_commands -eq 1 ]; then
            echo "$command"
        fi
        eval "$command" &
        children+=($!)
    done
    for child in "${children[@]}"; do
        wait $child;
        if [ $? -ne 0 ]; then
            _term;
        fi;
    done
    echo "Done"
    exit 0
fi
while [ $task -lt $nt ]; do
    task=$((task+1))
    if [ $protocol -eq 1 ]; then
        suffix="--suffix _proto1"
        short_suffix="_proto1"
        fixed_parameters="\"{\\\"duration_tau_inv\\\": null,\
                             \\\"duration_var0\\\": null,\
                             \\\"duration_background_var\\\": null,\
                             \\\"duration_background_mean\\\": null,\
                             \\\"duration_alpha\\\": null,\
                             \\\"duration_low_perror\\\": null,\
                             \\\"duration_high_perror\\\": null,\
                             \\\"intensity_tau_inv\\\": null,\
                             \\\"intensity_var0\\\": null,\
                             \\\"intensity_background_var\\\": null,\
                             \\\"intensity_background_mean\\\": null,\
                             \\\"intensity_alpha\\\": null,\
                             \\\"intensity_low_perror\\\": null,\
                             \\\"intensity_high_perror\\\": null,\
                             \\\"adaptation_amplitude\\\": null,\
                             \\\"adaptation_baseline\\\": null,\
                             \\\"adaptation_tau_inv\\\": null\
                            }\"
                         "
        start_from_fit_output="\"{\\\"method\\\": \\\"disc_$merit\\\",\
                                  \\\"optimizer\\\": \\\"$optimizer\\\",\
                                  \\\"suffix\\\": \\\"$short_suffix\\\"\
                                 }\"
                              "
        command1="$python fitter.py -e $experiment --method disc_$merit -t $task -nt $nt --save_plot_handler $verbosity $override \
            $suffix -o $optimizer"
        command2="$python fitter.py -e $experiment --method slider_$merit -t $task -nt $nt --save_plot_handler $verbosity $override \
            --fixed_parameters $fixed_parameters \
            --start_point_from_fit_output $start_from_fit_output \
            $suffix -o $optimizer"
        command1=$(clean_comm_whitespaces "$command1")
        command2=$(clean_comm_whitespaces "$command2")
        if [ $display_commands -eq 1 ]; then
            echo "$command1 && $command2"
        fi
        eval "$command1" && eval "$command2" &
        children+=($!)
        
        start_from_fit_output="\"{\\\"method\\\": \\\"disc_2alpha_$merit\\\",\
                                  \\\"optimizer\\\": \\\"$optimizer\\\",\
                                  \\\"suffix\\\": \\\"$short_suffix\\\"\
                                 }\"
                              "
        command1="$python fitter.py -e $experiment --method disc_2alpha_$merit -t $task -nt $nt --save_plot_handler $verbosity $override \
            $suffix -o $optimizer"
        command2="$python fitter.py -e $experiment --method slider_2alpha_$merit -t $task -nt $nt --save_plot_handler $verbosity $override \
            --fixed_parameters $fixed_parameters \
            --start_point_from_fit_output $start_from_fit_output \
            $suffix -o $optimizer"
        command1=$(clean_comm_whitespaces "$command1")
        command2=$(clean_comm_whitespaces "$command2")
        if [ $display_commands -eq 1 ]; then
            echo "$command1 && $command2"
        fi
        eval "$command1" && eval "$command2" &
        children+=($!)
    elif [ $protocol -eq 2 ]; then
        suffix="--suffix _proto2"
        short_suffix="_proto2"
        fixed_parameters="\"{\\\"duration_tau_inv\\\": null,\
                             \\\"duration_var0\\\": null,\
                             \\\"duration_x0\\\": null,\
                             \\\"duration_leak\\\": null,\
                             \\\"duration_background_mean\\\": null,\
                             \\\"duration_alpha\\\": null,\
                             \\\"intensity_tau_inv\\\": null,\
                             \\\"intensity_var0\\\": null,\
                             \\\"intensity_x0\\\": null,\
                             \\\"intensity_leak\\\": null,\
                             \\\"intensity_background_mean\\\": null,\
                             \\\"intensity_alpha\\\": null,\
                             \\\"adaptation_amplitude\\\": null,\
                             \\\"adaptation_baseline\\\": null,\
                             \\\"adaptation_tau_inv\\\": null\
                            }\"
                         "
        start_from_fit_output="\"{\\\"method\\\": \\\"slider_$merit\\\",\
                                  \\\"optimizer\\\": \\\"$optimizer\\\",\
                                  \\\"suffix\\\": \\\"$short_suffix\\\"\
                                 }\"
                              "
        command1="$python fitter.py -e $experiment --method slider_$merit -t $task -nt $nt --save_plot_handler $verbosity $override \
            $suffix -o $optimizer"
        command2="$python fitter.py -e $experiment --method disc_$merit -t $task -nt $nt --save_plot_handler $verbosity $override \
            --fixed_parameters $fixed_parameters \
            --start_point_from_fit_output $start_from_fit_output \
            $suffix -o $optimizer"
        if [ $display_commands -eq 1 ]; then
            echo "$command1 && $command2"
        fi
        command1=$(clean_comm_whitespaces "$command1")
        command2=$(clean_comm_whitespaces "$command2")
        eval "$command1" && eval "$command2" &
        children+=($!)
        
        start_from_fit_output="\"{\\\"method\\\": \\\"slider_2alpha_$merit\\\",\
                                  \\\"optimizer\\\": \\\"$optimizer\\\",\
                                  \\\"suffix\\\": \\\"$short_suffix\\\"\
                                 }\"
                              "
        command1="$python fitter.py -e $experiment --method slider_2alpha_$merit -t $task -nt $nt --save_plot_handler $verbosity $override \
            $suffix -o $optimizer"
        command2="$python fitter.py -e $experiment --method disc_2alpha_$merit -t $task -nt $nt --save_plot_handler $verbosity $override \
            --fixed_parameters $fixed_parameters \
            --start_point_from_fit_output $start_from_fit_output \
            $suffix -o $optimizer"
        command1=$(clean_comm_whitespaces "$command1")
        command2=$(clean_comm_whitespaces "$command2")
        if [ $display_commands -eq 1 ]; then
            echo "$command1 && $command2"
        fi
        eval "$command1" && eval "$command2" &
        children+=($!)
    elif [ $protocol -eq 3 ]; then
        suffix="--suffix _proto3"
        short_suffix="_proto3"
        fixed_parameters="\"{\\\"duration_tau_inv\\\": null,\
                             \\\"duration_var0\\\": null,\
                             \\\"duration_background_var\\\": null,\
                             \\\"duration_background_mean\\\": null,\
                             \\\"duration_alpha\\\": null,\
                             \\\"duration_low_perror\\\": null,\
                             \\\"duration_high_perror\\\": null,\
                             \\\"intensity_tau_inv\\\": null,\
                             \\\"intensity_var0\\\": null,\
                             \\\"intensity_background_var\\\": null,\
                             \\\"intensity_background_mean\\\": null,\
                             \\\"intensity_alpha\\\": null,\
                             \\\"intensity_low_perror\\\": null,\
                             \\\"intensity_high_perror\\\": null,\
                             \\\"adaptation_amplitude\\\": null,\
                             \\\"adaptation_baseline\\\": null,\
                             \\\"adaptation_tau_inv\\\": null\
                            }\"
                         "
        start_from_fit_output="\"{\\\"method\\\": \\\"disc_nll\\\",\
                                  \\\"optimizer\\\": \\\"$optimizer\\\",\
                                  \\\"suffix\\\": \\\"$short_suffix\\\"\
                                 }\"
                              "
        command1="$python fitter.py -e $experiment --method disc_nll -t $task -nt $nt --save_plot_handler $verbosity $override \
            $suffix -o $optimizer"
        command2="$python fitter.py -e $experiment --method slider_ls -t $task -nt $nt --save_plot_handler $verbosity $override \
            --fixed_parameters $fixed_parameters \
            --start_point_from_fit_output $start_from_fit_output \
            $suffix -o $optimizer"
        command1=$(clean_comm_whitespaces "$command1")
        command2=$(clean_comm_whitespaces "$command2")
        if [ $display_commands -eq 1 ]; then
            echo "$command1 && $command2"
        fi
        eval "$command1" && eval "$command2" &
        children+=($!)
        
        start_from_fit_output="\"{\\\"method\\\": \\\"disc_2alpha_nll\\\",\
                                  \\\"optimizer\\\": \\\"$optimizer\\\",\
                                  \\\"suffix\\\": \\\"$short_suffix\\\"\
                                 }\"
                              "
        command1="$python fitter.py -e $experiment --method disc_2alpha_nll -t $task -nt $nt --save_plot_handler $verbosity $override \
            $suffix -o $optimizer"
        command2="$python fitter.py -e $experiment --method slider_2alpha_ls -t $task -nt $nt --save_plot_handler $verbosity $override \
            --fixed_parameters $fixed_parameters \
            --start_point_from_fit_output $start_from_fit_output \
            $suffix -o $optimizer"
        command1=$(clean_comm_whitespaces "$command1")
        command2=$(clean_comm_whitespaces "$command2")
        if [ $display_commands -eq 1 ]; then
            echo "$command1 && $command2"
        fi
        eval "$command1" && eval "$command2" &
        children+=($!)
    else
        suffix=""
        short_suffix=""
        if [ "$subjects" != "rats" ]; then
            if [ $display_commands -eq 1 ]; then
                echo "$python fitter.py -e $experiment --method ""$exps""_""$merit"" -t $task -nt $nt --save_plot_handler $suffix $verbosity $override -o $optimizer"
            fi
            $python fitter.py -e $experiment --method "$exps"_"$merit" -t $task -nt $nt --save_plot_handler $suffix $verbosity $override -o $optimizer &
            children+=($!)
        fi
        if [ $display_commands -eq 1 ]; then
            echo "$python fitter.py -e $experiment --method ""$exps""_2alpha_""$merit"" -t $task -nt $nt --save_plot_handler $suffix $verbosity $override -o $optimizer"
        fi
        $python fitter.py -e $experiment --method "$exps"_2alpha_"$merit" -t $task -nt $nt --save_plot_handler $suffix $verbosity $override -o $optimizer &
        children+=($!)
    fi
done

for child in "${children[@]}"; do
    wait $child;
    if [ $? -ne 0 ]; then
        _term;
    fi;
done

#~ trap - SIGINT SIGQUIT SIGABRT SIGKILL SIGTERM ERR
trap - SIGINT SIGQUIT SIGABRT SIGKILL SIGTERM

temp1=figs/temp1_"$$".pdf
temp2=figs/temp2_"$$".pdf

if [ "$subjects" != "rats" ]; then
    if [ $display_commands -eq 1 ]; then
        echo "$python fitter.py -e $experiment --method ""$exps""_""$merit"" --no-fit --load_plot_handler --save $verbosity $override $suffix -o $optimizer&& 
        mv figs/fits_""$experiment""_""$exps""_""$merit""""$short_suffix"".pdf $temp2 && 
        $python fitter.py -e $experiment --method ""$exps""_""$merit"" --no-fit --load_plot_handler --save --plot_merge all $verbosity $override $suffix -o $optimizer&& 
        mv figs/fits_""$experiment""_""$exps""_""$merit""""$short_suffix"".pdf $temp1 && 
        pdftk $temp1 $temp2 cat output figs/""$experiment""_""$exps""_single_alpha""$short_suffix"".pdf && 
        rm $temp1 $temp2 && 
        $python fitter.py -e $experiment --method ""$exps""_2alpha_""$merit"" --no-fit --load_plot_handler --save $verbosity $override $suffix -o $optimizer&& 
        mv figs/fits_""$experiment""_""$exps""_2alpha_""$merit""""$short_suffix"".pdf $temp2 && 
        $python fitter.py -e $experiment --method ""$exps""_2alpha_""$merit"" --no-fit --load_plot_handler --save --plot_merge all $verbosity $override $suffix -o $optimizer&& 
        mv figs/fits_""$experiment""_""$exps""_2alpha_""$merit""""$short_suffix"".pdf $temp1 && 
        pdftk $temp1 $temp2 cat output figs/""$experiment""_""$exps""_2alphas""$short_suffix"".pdf && 
        rm $temp1 $temp2";
    fi
    echo Constructing plots for "$experiment"_"$exps"_"$merit""$short_suffix" with for subject separately && \
    $python fitter.py -e $experiment --method "$exps"_"$merit" --no-fit --load_plot_handler --save $verbosity $override $suffix -o $optimizer&& \
    mv figs/fits_"$experiment"_"$exps"_"$merit""$short_suffix".pdf $temp2 && \
    echo Constructing plots for "$experiment"_"$exps"_"$merit""$short_suffix" with all subjects merged && \
    $python fitter.py -e $experiment --method "$exps"_"$merit" --no-fit --load_plot_handler --save --plot_merge all $verbosity $override $suffix -o $optimizer&& \
    mv figs/fits_"$experiment"_"$exps"_"$merit""$short_suffix".pdf $temp1 && \
    echo "Joining pdfs" && \
    pdftk $temp1 $temp2 cat output figs/"$experiment"_"$exps"_single_alpha_"$merit""$short_suffix".pdf && \
    rm $temp1 $temp2 && \
    \
    echo Constructing plots for "$experiment"_"$exps"_2alpha_"$merit""$short_suffix" for each subject separately && \
    $python fitter.py -e $experiment --method "$exps"_2alpha_"$merit" --no-fit --load_plot_handler --save $verbosity $override $suffix -o $optimizer&& \
    mv figs/fits_"$experiment"_"$exps"_2alpha_"$merit""$short_suffix".pdf $temp2 && \
    echo Constructing plots for "$experiment"_"$exps"_2alpha_"$merit""$short_suffix" with all subjects merged && \
    $python fitter.py -e $experiment --method "$exps"_2alpha_"$merit" --no-fit --load_plot_handler --save --plot_merge all $verbosity $override $suffix -o $optimizer&& \
    mv figs/fits_"$experiment"_"$exps"_2alpha_"$merit""$short_suffix".pdf $temp1 && \
    echo "Joining pdfs" && \
    pdftk $temp1 $temp2 cat output figs/"$experiment"_"$exps"_2alphas_"$merit""$short_suffix".pdf && \
    rm $temp1 $temp2;
else
    if [ $display_commands -eq 1 ]; then
        echo "$python fitter.py -e $experiment --method ""$exps""_2alpha_""$merit"" --no-fit --load_plot_handler --save $verbosity $override $suffix -o $optimizer&& 
        mv figs/fits_""$experiment""_""$exps""_2alpha_""$merit""""$short_suffix"".pdf $temp2 && 
        $python fitter.py -e $experiment --method ""$exps""_2alpha_""$merit"" --no-fit --load_plot_handler --save --plot_merge all $verbosity $override $suffix -o $optimizer&& 
        mv figs/fits_""$experiment""_""$exps""_2alpha_""$merit""""$short_suffix"".pdf $temp1 && 
        pdftk $temp1 $temp2 cat output figs/""$experiment""_""$exps""_2alphas""$short_suffix"".pdf && 
        rm $temp1 $temp2";
    fi
    echo Constructing plots for "$experiment"_"$exps"_2alpha_"$merit""$short_suffix" for each subject separately && \
    $python fitter.py -e $experiment --method "$exps"_2alpha_"$merit" --no-fit --load_plot_handler --save $verbosity $override $suffix -o $optimizer&& \
    mv figs/fits_"$experiment"_"$exps"_2alpha_"$merit""$short_suffix".pdf $temp2 && \
    echo Constructing plots for "$experiment"_"$exps"_2alpha_"$merit""$short_suffix" with all subjects merged && \
    $python fitter.py -e $experiment --method "$exps"_2alpha_"$merit" --no-fit --load_plot_handler --save --plot_merge all $verbosity $override $suffix -o $optimizer&& \
    mv figs/fits_"$experiment"_"$exps"_2alpha_"$merit""$short_suffix".pdf $temp1 && \
    echo "Joining pdfs" && \
    pdftk $temp1 $temp2 cat output figs/"$experiment"_"$exps"_2alphas_"$merit""$short_suffix".pdf && \
    rm $temp1 $temp2;
fi

echo "Done"
exit 0
