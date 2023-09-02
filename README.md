# FSRS vs SM-17
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

It is a simple comparsion between FSRS and SM-17. [FSRS-v-SM16-v-SM17.ipynb](./FSRS-v-SM16-v-SM17.ipynb) is the notebook for the comparsion.

Due to the difference between the workflow of SuperMemo and Anki, it is not easy to compare the two algorithms. I tried to make the comparison as fair as possible. Here is some notes:
- The first interval in SuperMemo is the duration between creating the card and the first review. In Anki, the first interval is the duration between the first review and the second review. So I removed the first record of each card in SM-17 data.
- There are six grades in SuperMemo, but only four grades in Anki. So I merged 0, 1 and 2 in SuperMemo to 1 in Anki, and mapped 3, 4, and 5 in SuperMemo to 2, 3, and 4 in Anki.
- I use the `R (SM17)(exp)` recorded in `sm18/systems/{collection_name}/stats/SM16-v-SM17.csv` as the prediction of SM-17. Reference: [Confusion among R(SM16), R(SM17)(exp), R(SM17), R est. and expFI.](https://supermemopedia.com/wiki/Confusion_among_R(SM16),_R(SM17)(exp),_R(SM17),_R_est._and_expFI.)
- To ensure FSRS has the same information as SM-17, I implement an [online learning](https://en.wikipedia.org/wiki/Online_machine_learning) version of FSRS, where FSRS has zero knowledge of the future reviews as SM-17 does.
- The results are based on the data from a small group of people. It may be different from the result of other SuperMemo users.

## Result

Total users: 6

Total repetitions: 92,499

| Algorithm | Log Loss | RMSE | Universal Metric |
| --- | --- | --- | --- |
| FSRS | 0.3368 | 0.3115 | 0.0359 |
| SM-17 | 0.3926 | 0.3231 | 0.0475 |
| SM-16 | 0.3780 | 0.3296 | 0.0524 |

Smaller is better.

## Share your data

If you would like to support this project, please consider sharing your data with us. The shared data will be stored in [./dataset/](./dataset/) folder. 

You can open an issue to submit it: https://github.com/open-spaced-repetition/fsrs-vs-sm17/issues/new/choose


## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/leee-z"><img src="https://avatars.githubusercontent.com/u/48952110?v=4?s=100" width="100px;" alt="leee_"/><br /><sub><b>leee_</b></sub></a><br /><a href="#data-leee-z" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.zhihu.com/people/L.M.Sherlock"><img src="https://avatars.githubusercontent.com/u/32575846?v=4?s=100" width="100px;" alt="Jarrett Ye"/><br /><sub><b>Jarrett Ye</b></sub></a><br /><a href="#data-L-M-Sherlock" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Shore3145"><img src="https://avatars.githubusercontent.com/u/106439025?v=4?s=100" width="100px;" alt="天空守望者"/><br /><sub><b>天空守望者</b></sub></a><br /><a href="#data-Shore3145" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/reallyyy"><img src="https://avatars.githubusercontent.com/u/39750041?v=4?s=100" width="100px;" alt="reallyyy"/><br /><sub><b>reallyyy</b></sub></a><br /><a href="#data-reallyyy" title="Data">🔣</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->