// Twilight Fable Studio catalog (`gallery/catalog.js`): the one style this repo ships.
export const STYLES = [
  {
    id: 'twilight-fable',
    name: 'Twilight Bioluminescent Fable',
    zh: '暮光柔焦绘本',
    scenes: 4,
    dur: 14.0,
    preset: 'twilight',
    stageMode: '3d-pure',
    defaultOverlay: 'none',
    kernel: './kernel.js',
    bestFor: [
      '治愈绘本童话 / 奇幻森林冒险 / 萌宠与自然生灵温情故事',
      '星空许愿 / 仲夏夜萤火 / 极光漫游 / 梦境与情绪疗愈短片',
      '香氛精油 / 睡眠助眠 / 茶饮花果 / 温暖灯光与家居美学',
      '音乐MV视觉 / 节气夜游 / 绘本风科普与诗意寓言',
    ],
    avoidFor: [
      '高密度现代金融K线与财务报表',
      '纯命令行代码终端演示',
      '长吻长腿写实动物（鹿/马）的面部特写与主角戏（尚无 sleek 体型；仅可作中远景 spirit 灵影）',
    ],
    keywords: [
      '暮光', '柔焦', '散景', '绘本', '星光', '星空', '萤火', '萤火虫', '蒲公英', '极光',
      '森林', '雾林', '树洞', '暖窗', '浅滩', '溪流', '水波', '倒影', '灵兽', '萌宠',
      '小猫', '小熊', '小狐狸', '耳廓狐', '长耳兔', '小鹿', '蘑菇', '发光', '荧光', '夜空',
      '梦境', '睡眠', '晚安', '助眠', '香氛', '治愈', '唯美', '温暖', '许愿', '星河', '童话',
      'twilight', 'fable', 'bokeh', 'bioluminescent', 'starlight', 'aurora', 'firefly',
      'dandelion', 'dreamy', 'healing', 'storybook', 'lullaby', 'critter', 'kitten', 'bear', 'fox', 'bunny',
    ],
    cameraPresets: [
      'macroPush',
      'widePull',
      'craneReveal',
      'lateralTrack',
      'finaleOrbit',
      'tabletopOrbit',
    ],
  },
];

export const SIZES = [
  { id: '1920x1080', name: '16:9 横屏' },
  { id: '1080x1920', name: '9:16 竖屏' },
  { id: '1080x1080', name: '1:1 方画幅' },
];
