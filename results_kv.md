(ssm_hw6) [cs601-asures13@gpuz01 kv-quantization]$ python generate.py --checkpoint_path checkpoints/Mistral-7B/model.pth --prompt "Once upon a time" --max_new_tokens 512 --compress_kv --window_size 128 --sink_size 32
Using device=cuda
Loading model ...
[DEBUG] Model loaded successfully.
[DEBUG] Model config: ModelArgs(block_size=2048, vocab_size=32768, n_layer=32, n_head=32, dim=4096, intermediate_size=14336, n_local_heads=8, head_dim=128, rope_base=10000, norm_eps=1e-05, rope_scaling=None, has_qkv_bias=False)
[DEBUG] Model total parameters: 7248.02M
Time to load model: 4.62 seconds
[DEBUG] Tokenizer loaded successfully.
[DEBUG] Prompt encoded, shape: torch.Size([5]), first 10 tokens: [1, 6481, 4482, 1032, 1495]
[DEBUG] Entered generate()
[DEBUG] prompt shape: torch.Size([5])
[DEBUG] batch_size = 1, max_new_tokens = 512
[DEBUG] Calling prefill()
[DEBUG] Inside prefill()
[DEBUG] x.shape: torch.Size([1, 5]), input_pos.shape: torch.Size([5])
[DEBUG] After view(-1), input_pos.shape: torch.Size([5])
[DEBUG] Prefill completed, next token sampled.
[DEBUG] Starting decode_n_tokens() loop
Once upon a time I used to work from 8am to 6pm… I used to be so tired I literally used to fall asleep in the shower… which is why I always used to shower at night… It was a lot more convenient that way…

Anyways I used to pretty much spend all of my time at work… I would spend all of my time at work… so, therefore, all of my time during those days was spent at work…

So, when I got laid off from work and then, a few months later, I started working at McDonald’s as a cashier… Well, you can’t really work at McDonald’s as a cashier if you don’t have a timecard… Well, you can’t really work at McDonald’s as a cashier if you don’t have a timecard…

So, therefore, all of my time during those days was spent working at McDonald’s…́ 8:00am to 6:00pm…́ 8:00am to 6:00pm…

And then, when I got home after working at McDonald’s as a cashier in the morning from 8:00am to 6:00pm…

Once upon a time… I used to work from 8am to 6pm…

I used to be so tired I literally used to fall asleep in the evening … from 8:00pm to 6:00pm…

And then, when I got home after working at McDonald’s as a cashier in the evening from 8:00pm to 6:00pm…

Once upon a time… I used to work from 8am to 6pm… I used to be so tired I literally used to fall asleep in the evening… from 8:00pm to 6:00pm…

And then, when I got home after working at McDonald’s as a cashier in the evening… from 8:00pm to 6:00pm…

I would see “I’m tired of always working so hard and not getting anywhere… I’m tired of always being broke and never being able to do anything because I don’t have any money… I’m tired of always having to worry about things because I don’t have enough money… I’m tired
Time for inference 1: 100.45 sec total, 5.10 tokens/sec
Bandwidth achieved: 72.52 GB/s
FLOPS achieved: 0.07 TF/s

[DEBUG] Entered generate()
[DEBUG] prompt shape: torch.Size([5])
[DEBUG] batch_size = 1, max_new_tokens = 512
[DEBUG] Calling prefill()
[DEBUG] Inside prefill()
[DEBUG] x.shape: torch.Size([1, 5]), input_pos.shape: torch.Size([5])
[DEBUG] After view(-1), input_pos.shape: torch.Size([5])
[DEBUG] Prefill completed, next token sampled.
[DEBUG] Starting decode_n_tokens() loop
Once upon a time, there lived a great, wise, and powerful monarch. He was a man who knew what he wanted out of life. His one dream was to one day rule the world. He was a man of great ambition.

The King had many servants at his beck and call. He had a swashbuckling sword master, a strong and sturdy stonecutter, a reliable and trustworthy tax collector, a brilliant and wise wizard, and a notorious and infamous highwayman.

The King was not a man to give up on his dreams. He knew that if he wanted to achieve his goal of becoming the ruler of the entire world, then he was going to have to put forth a tremendous amount of effort.

The King had a plan. He was going to gather all of his servants together, and he was going to send them out on a journey. He was going to task them with the difficult and arduous task of collecting the world’s seven treasures.

The seven treasures that the King had instructed his servants to go out into the world and collect, were the following:

1. The Flame of Desire
2. The Mirror of the Soul
3. The Silver Mirror of the Heart
4. The Silver Mirror of the Soul
5. The Silver Mirror of the Mind
6. The Silver Mirror of the Spirit

The Flame of Desire

The first treasure was the Flame of Desire. This was a flame that was said to contain the deepest desires of one’s heart.

The Mirror of the Soul

The second treasure was the Mirror of the Soul. This was a mirror that was said to contain the deepest truths about one’s soul.

The Silver Mirror of the Heart

The third treasure was the Silver Mirror of the Heart. This was a mirror that was said to contain the deepest and most beautiful truths about one’s heart.

The Silver Mirror of the Soul

The fourth treasure was the Silver Mirror of the Soul. This was a mirror that was said to contain the deepest and most beautiful truths about one’s soul.

The Silver Mirror of the Heart

The fifth treasure was the Silver Mirror of the Heart. This was a mirror that was said to contain the deepest and most beautiful truths about
Time for inference 2: 96.79 sec total, 5.29 tokens/sec
Bandwidth achieved: 75.27 GB/s
FLOPS achieved: 0.08 TF/s

[DEBUG] Entered generate()
[DEBUG] prompt shape: torch.Size([5])
[DEBUG] batch_size = 1, max_new_tokens = 512
[DEBUG] Calling prefill()
[DEBUG] Inside prefill()
[DEBUG] x.shape: torch.Size([1, 5]), input_pos.shape: torch.Size([5])
[DEBUG] After view(-1), input_pos.shape: torch.Size([5])
[DEBUG] Prefill completed, next token sampled.
[DEBUG] Starting decode_n_tokens() loop
Once upon a time there was a man named David. He was a shepherd. He had a relationship with the sheep. He was close to each one individually. He knew what they were doing, who they were with, and what their needs were at any given point in time.

David also had another unique ability. He could easily leave the sheep behind and walk into the wilderness and fight a lion, or run after a bear, or capture a predator.

And when he returned back to the sheep, the sheep were grateful and they all gathered around his neck and kissed him.

The End

A.J. Castellitto

www.truthinsomnia.com ISBN 978-1-93519-4-1 ISBN 978-1-93519-4-7 ISBN 978-1-93519-4-5 ISBN 978-1-93519-4-8 ISBN 978-1-93519-4-7 ISBN 978-1-93519-4-9 ISBN 978-1-93519-4-2 ISBN 978-1-93519-4-6 ISBN 978-1-93519-4-3 ISBN 978-1-93519-4-4 ISBN 978-1-93519-4-5 ISBN 978-1-93519-4-1 ISBN 0-85229-00-2 ISBN 978-0-85229-00-2

## 3. The Creation of the Lords of the Manor

Once upon a time there was a man named Henry. He was a king. He had the power to rule over a country.

King Henry was also a man of God. He was a believer in the One True God, the Creator of the universe and all that is in it.

King Henry was also a man of science. He was a believer in the Laws of Nature, the rules that govern how the world works.

King Henry was also a man of the people
Time for inference 3: 96.67 sec total, 5.30 tokens/sec
Bandwidth achieved: 75.35 GB/s
FLOPS achieved: 0.08 TF/s

[DEBUG] Entered generate()
[DEBUG] prompt shape: torch.Size([5])
[DEBUG] batch_size = 1, max_new_tokens = 512
[DEBUG] Calling prefill()
[DEBUG] Inside prefill()
[DEBUG] x.shape: torch.Size([1, 5]), input_pos.shape: torch.Size([5])
[DEBUG] After view(-1), input_pos.shape: torch.Size([5])
[DEBUG] Prefill completed, next token sampled.
[DEBUG] Starting decode_n_tokens() loop
Once upon a time, there lived a little girl.

She was afraid.

She had just heard a story that involved dragons. They were huge and scary and they wanted to eat ALL THE PEOPLE.

She was afraid.

She was concerned that the story would come true. She didn’t want to live in fear, and she didn’t want to live in a world where dragons posed a real and present threat to people.

She looked and looked and looked and she couldn’t find any dragons to be afraid of.

She realized that she can’t be afraid of that which does not exist.

The dragons were imaginary. The dragon story was imaginary. The world in which the dragon story took place was imaginary.

There were no real life dragons.

There was no real world dragon story.

There was no real world where the dragon story took place.

The only dragon story, and the only world in which the dragon story took place, was an imaginary one.

The only dragons, and the only dragon story, were virtual ones.

Once upon a time, there did not exist a real dragon, or a real dragon story, or a real world in which a real dragon story took place.

The only real dragon, and the only real dragon story, and the only real world in which a real dragon story took place, were virtual ones.

Once upon a time, there lived a little girl who was afraid.

She was afraid because she had just heard a story that involved dragons. They were the real ones.

They were also the only real dragons, and the only real dragon story, and the only real world in which a real dragon story took place.

They were also the only real dragons who ever lived in the real world in which a real dragon story took place.

They were also the only real dragons who ever lived in the real world in which a real dragon story took place.

They were also the only real dragons who ever lived in the real world in which a real dragon story took place.

They were also the only real dragons who ever lived in the real world in which a real dragon story took place.

They were also the only real dragons who ever lived in the real world in which a real dragon story took place.

They were also the only real dragons
Time for inference 4: 96.22 sec total, 5.32 tokens/sec
Bandwidth achieved: 75.71 GB/s
FLOPS achieved: 0.08 TF/s

[DEBUG] Entered generate()
[DEBUG] prompt shape: torch.Size([5])
[DEBUG] batch_size = 1, max_new_tokens = 512
[DEBUG] Calling prefill()
[DEBUG] Inside prefill()
[DEBUG] x.shape: torch.Size([1, 5]), input_pos.shape: torch.Size([5])
[DEBUG] After view(-1), input_pos.shape: torch.Size([5])
[DEBUG] Prefill completed, next token sampled.
[DEBUG] Starting decode_n_tokens() loop
Once upon a time, a young man named Adam came to work at Aspen Insurance.  He was young, eager and determined to prove his worth.

On his first day, Adam was assigned to a desk near the bustling telecommunications room.  Despite the noise, Adam was determined to do his best.

By the end of his first week, Adam had impressed his supervisors with his ability to make sense of complicated insurance policies.

As Adam continued to work hard, he soon found himself promoted to a more important position.

Over time, Adam’s hard work and dedication paid off.  He was promoted to management and was eventually chosen to be the president of the company.

Adam’s story is a classic example of how hard work and dedication can pay off.  Adam’s story is a reminder that if you work hard and stay dedicated to your goals, success will eventually come your way.мрдщшщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщщшки...

...and they lived happily ever after.

--

The Man
Time for inference 5: 96.97 sec total, 5.28 tokens/sec
Bandwidth achieved: 75.12 GB/s
FLOPS achieved: 0.08 TF/s

==========
Batch Size: 1
Prompt Length: 5
Generated tokens: 512
Average tokens/sec: 5.26
Memory used: 14.64 GB