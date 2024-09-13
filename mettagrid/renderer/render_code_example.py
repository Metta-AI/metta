
# def draw_bars(rl, entity, x, y, width, height=4, draw_text=False):
#     health_bar = entity.health / entity.max_health
#     mana_bar = entity.mana / entity.max_mana
#     if entity.max_health == 0:
#         health_bar = 2
#     if entity.max_mana == 0:
#         mana_bar = 2
#     rl.DrawRectangle(x, y, width, height, [255, 0, 0, 255])
#     rl.DrawRectangle(x, y, int(width*health_bar), height, [0, 255, 0, 255])

#     if entity.entity_type == 0:
#         rl.DrawRectangle(x, y - height - 2, width, height, [255, 0, 0, 255])
#         rl.DrawRectangle(x, y - height - 2, int(width*mana_bar), height, [0, 255, 255, 255])

#     if draw_text:
#         health = int(entity.health)
#         mana = int(entity.mana)
#         max_health = int(entity.max_health)
#         max_mana = int(entity.max_mana)
#         rl.DrawText(f'Health: {health}/{max_health}'.encode(),
#             x+8, y+2, 20, [255, 255, 255, 255])
#         rl.DrawText(f'Mana: {mana}/{max_mana}'.encode(),
#             x+8, y+2 - height - 2, 20, [255, 255, 255, 255])

#         #rl.DrawRectangle(x, y - 2*height - 4, int(width*mana_bar), height, [255, 255, 0, 255])
#         rl.DrawText(f'Experience: {entity.xp}'.encode(),
#             x+8, y - 2*height - 4, 20, [255, 255, 255, 255])

#     elif entity.entity_type == 0:
#         rl.DrawText(f'Level: {entity.level}'.encode(),
#             x+4, y -2*height - 12, 12, [255, 255, 255, 255])



        # Draw HUD
        # player = entities[0]
        # hud_y = self.height*ts - 2*ts
        # draw_bars(rl, player, 2*ts, hud_y, 10*ts, 24, draw_text=True)

        # off_color = [255, 255, 255, 255]
        # on_color = [0, 255, 0, 255]

        # q_color = on_color if skill_q else off_color
        # w_color = on_color if skill_w else off_color
        # e_color = on_color if skill_e else off_color

        # q_cd = player.q_timer
        # w_cd = player.w_timer
        # e_cd = player.e_timer

        # rl.DrawText(f'Q: {q_cd}'.encode(), 13*ts, hud_y - 20, 40, q_color)
        # rl.DrawText(f'W: {w_cd}'.encode(), 17*ts, hud_y - 20, 40, w_color)
        # rl.DrawText(f'E: {e_cd}'.encode(), 21*ts, hud_y - 20, 40, e_color)
        # rl.DrawText(f'Stun: {player.stun_timer}'.encode(), 25*ts, hud_y - 20, 20, e_color)
        # rl.DrawText(f'Move: {player.move_timer}'.encode(), 25*ts, hud_y, 20, e_color)
